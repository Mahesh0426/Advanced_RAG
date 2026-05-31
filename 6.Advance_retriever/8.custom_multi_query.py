# =============================================================================
# Custom Multi-Query Retriever — Built from scratch using BaseRetriever
# =============================================================================
# LangChain's built-in MultiQueryRetriever is convenient but a black box.
# This file builds the same concept manually by subclassing BaseRetriever,
# giving full control over every step:
#
#   1. Prompt design  — exactly how the LLM is instructed to rephrase
#   2. Structured output — parsed via Pydantic (no fragile string splitting)
#   3. Retrieval loop  — how each query variant hits the vector store
#   4. Deduplication   — how repeated chunks are removed before returning
#
# Understanding this custom version makes it much easier to extend later,
# e.g. adding query scoring, per-variant k values, or async retrieval.
# =============================================================================

from dotenv import load_dotenv
from typing import Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# Load OPENAI_API_KEY from .env
load_dotenv()

# ── Embedding & LLM setup ────────────────────────────────────────────────────

# Converts text into dense vectors for semantic similarity search
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Used exclusively for generating alternative query phrasings,
# NOT for producing the final answer.
# temperature=0.3 — slight creativity so variants are meaningfully different
# from each other, not near-identical paraphrases
llm = ChatOpenAI(model="gpt-5-mini", temperature=0.3)

# ── Sample documents ─────────────────────────────────────────────────────────

docs = [
    Document(
        page_content=(
            "Biotechnology companies are developing novel protein-based therapies that target specific "
            "disease pathways with unprecedented precision. Synthetic biology techniques allow scientists "
            "to engineer microorganisms that produce pharmaceutical compounds at industrial scale. "
            "Bioreactor technologies have dramatically reduced the cost of producing monoclonal antibodies, "
            "making treatments for autoimmune diseases and cancers more accessible. Microbiome research is "
            "revealing how manipulating gut bacteria can influence everything from mental health to "
            "metabolic disorders."
        ),
        metadata={"topic": "biotechnology"},
    ),
    Document(
        page_content=(
            "Zero-trust architecture has become the gold standard for enterprise network security, "
            "requiring continuous verification rather than relying on perimeter defenses. Machine learning "
            "models now detect anomalous network behavior in real time, reducing the window between "
            "intrusion and detection from months to minutes. Ransomware attacks on critical infrastructure "
            "have forced governments to establish mandatory incident reporting requirements for healthcare "
            "and energy sectors. Post-quantum cryptography standards are being finalized to protect "
            "sensitive data against future quantum computing threats."
        ),
        metadata={"topic": "cybersecurity"},
    ),
    Document(
        page_content=(
            "Brain-computer interfaces are enabling paralyzed patients to control prosthetic limbs and "
            "communicate using only their neural signals. Optogenetics allows researchers to activate or "
            "silence specific neuron populations with light, accelerating the understanding of neural "
            "circuit function and disease. Advanced neuroimaging techniques using fMRI and "
            "magnetoencephalography are mapping brain connectivity with millimeter precision, unlocking "
            "new treatments for depression and PTSD. Neurofeedback therapies are showing promise for "
            "cognitive rehabilitation following traumatic brain injuries."
        ),
        metadata={"topic": "neuroscience"},
    ),
    Document(
        page_content=(
            "Perovskite solar cells have achieved efficiency ratings exceeding 33%, surpassing traditional "
            "silicon panels and promising dramatically lower manufacturing costs. Grid-scale battery "
            "storage using iron-air and sodium-ion technologies is making renewable energy dispatchable "
            "around the clock without relying on rare earth metals. Offshore floating wind farms are "
            "expanding into deep-water regions previously inaccessible to fixed-foundation turbines, "
            "multiplying available wind energy capacity. Green hydrogen produced via electrolysis is "
            "emerging as a critical energy carrier for decarbonizing heavy industry and long-haul "
            "transport."
        ),
        metadata={"topic": "renewable_energy"},
    ),
    Document(
        page_content=(
            "Surgical robots equipped with haptic feedback allow surgeons to perform minimally invasive "
            "procedures with sub-millimeter precision, reducing patient recovery times significantly. "
            "Collaborative robots in manufacturing now work safely alongside humans using advanced "
            "computer vision and force sensing, without the need for physical barriers. Autonomous mobile "
            "robots are transforming warehouse logistics, optimizing pick-and-place operations and "
            "reducing fulfillment errors. Soft robots inspired by biological organisms are being developed "
            "for delicate tasks in agriculture, search-and-rescue, and medical drug delivery."
        ),
        metadata={"topic": "robotics"},
    ),
    Document(
        page_content=(
            "Base editing and prime editing technologies offer more precise alternatives to CRISPR-Cas9, "
            "enabling single-letter corrections to the genome without creating double-strand breaks. "
            "Gene therapy trials using adeno-associated virus vectors have achieved functional cures for "
            "hemophilia B and spinal muscular atrophy. Epigenome editing tools allow researchers to "
            "switch genes on or off without altering the underlying DNA sequence, opening new avenues "
            "for treating complex diseases. Polygenic risk scoring combined with germline analysis is "
            "enabling predictive medicine that identifies disease susceptibility decades before symptoms "
            "appear."
        ),
        metadata={"topic": "genetic_engineering"},
    ),
]

print(f"Created {len(docs)} documents")

# ── Text splitting ────────────────────────────────────────────────────────────
# Split long documents into smaller chunks before embedding.
# chunk_size=300   → each chunk is at most 300 characters
# chunk_overlap=50 → adjacent chunks share 50 characters so sentences
#                    split across a boundary still appear in at least one chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(docs)

# ── Vector store & base retriever ────────────────────────────────────────────
# Embed all chunks and store them in memory (no external DB required).
# base_retriever wraps the store and returns the top-k most similar chunks
# for any query string passed to it.
# k=3 means 3 chunks per query variant — the custom retriever calls this
# once per generated variant, so total docs before dedup = variants × 3
vectorstore = InMemoryVectorStore.from_documents(docs, embedding=embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Structured output schema ──────────────────────────────────────────────────
# QueriesSchema is the structured output the LLM produces — a list of alternative questions.
# Using Pydantic here instead of plain string parsing means:
#   - The LLM's response is automatically validated against this shape
#   - No manual splitting on newlines or cleaning of bullet points
#   - If the LLM returns the wrong shape, Pydantic raises a clear error
class QueriesSchema(BaseModel):
    queries: list[str] = Field(description="List of 3 alternative versions of the question")


# ── Query-generation prompt ───────────────────────────────────────────────────
# This prompt instructs the LLM to rephrase the original question from
# multiple angles. The key insight: different phrasings embed to different
# vector positions, so each variant probes a different neighborhood of the
# vector space — catching documents the original phrasing would have missed.
prompt = ChatPromptTemplate.from_template(
    "You are an AI language model assistant. Your task is to generate 3 different versions of "
    "the given user question to retrieve relevant documents from a vector database. "
    "By generating multiple perspectives on the user question, your goal is to help the user "
    "overcome some of the limitations of distance-based similarity search. "
    "Provide these alternative questions separated by newlines.\n\n"
    "Original question: {question}"
)

# Bind QueriesSchema as the expected output shape.
# with_structured_output instructs the LLM to return JSON matching the schema
# and wraps the response in a validated QueriesSchema instance automatically.
llm_structured_output = llm.with_structured_output(QueriesSchema)

# Build the query-generation chain:
#   prompt → formats the template with {question}
#   llm_structured_output → calls the LLM and parses response into QueriesSchema
# with_structured_output binds QueriesSchema so the chain always returns a QueriesSchema instance
query_chain = prompt | llm_structured_output


# ── Custom retriever class ────────────────────────────────────────────────────
# Subclassing BaseRetriever is the standard LangChain extension point for
# custom retrieval logic. The only required method is _get_relevant_documents().
# Implementing it here (rather than using MultiQueryRetriever) lets us:
#   - Inspect generated queries before retrieval
#   - Swap in any base_retriever without changing the class
#   - Plug this retriever into any LangChain chain or agent transparently
class CustomMultiQueryRetriever(BaseRetriever):
    """Retriever that generates multiple query perspectives via an LLM and deduplicates results."""

    # The underlying retriever that does the actual vector similarity search.
    # Declared as a Pydantic field so LangChain can serialize/deserialize the chain.
    base_retriever: BaseRetriever

    # The full query-generation chain (prompt | llm.with_structured_output).
    # Typed as Any because LangChain LCEL chains don't have a single concrete type.
    query_chain: Any  # prompt | llm.with_structured_output(QueriesSchema)

    def _generate_queries(self, query: str) -> list[str]:
        # Invoke the chain: fills {question} in the prompt, calls the LLM,
        # and returns a fully validated QueriesSchema instance.
        # .queries gives us the plain list[str] we need for retrieval.
        result: QueriesSchema = self.query_chain.invoke({"question": query})
        return result.queries

    def _unique_documents(self, documents: list[Document]) -> list[Document]:
        # Deduplicate by page_content, preserving first-occurrence order.
        # Using a set for O(1) membership checks — important when variant × k
        # can produce dozens of candidate chunks.
        # Note: two chunks with identical text but different metadata are
        # treated as the same document here (content-based dedup).
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in documents:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique.append(doc)
        return unique

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # This is the single method BaseRetriever requires us to implement.
        # LangChain calls it whenever .invoke() or .get_relevant_documents() is used.

        # Step 1: Ask the LLM for N alternative phrasings of the original query.
        # Each phrasing explores a different semantic angle of the same intent.
        queries = self._generate_queries(query)

        # Step 2: Run each alternative query through the base vector retriever.
        # all_docs may contain duplicates if the same chunk scores highly for
        # multiple query variants — that's expected and handled in step 3.
        all_docs: list[Document] = []
        for q in queries:
            all_docs.extend(self.base_retriever.invoke(q))

        # Step 3: Remove duplicate chunks and return the clean union.
        # The final list covers more of the corpus than any single query would.
        return self._unique_documents(all_docs)


# ── Instantiate the custom retriever ─────────────────────────────────────────
# Pass in the two dependencies declared as Pydantic fields above.
# From here it behaves exactly like any other LangChain retriever.
retriever = CustomMultiQueryRetriever(base_retriever=base_retriever, query_chain=query_chain)

query = "How are modern technologies improving human health?"

# ── Preview generated query variants ─────────────────────────────────────────
# Invoke the query_chain directly (before full retrieval) so we can inspect
# what alternative questions the LLM produced for this input.
# This is purely diagnostic — the retriever will invoke the chain again internally.
parsed = query_chain.invoke({"question": query})

print("Generated alternative queries:")
for q in parsed.queries:
    print(f"  - {q}")
print()

# ── Run full multi-query retrieval ────────────────────────────────────────────
# .invoke() calls _get_relevant_documents() under the hood:
#   generate variants → retrieve per variant → deduplicate → return
results = retriever.invoke(query)

print(f"Retrieved {len(results)} unique documents:\n")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} [{doc.metadata.get('topic')}] ---")
    print(doc.page_content)
    print()