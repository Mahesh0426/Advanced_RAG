# =============================================================================
# Multi-Query Retriever — Advanced RAG
# =============================================================================
# Standard RAG retrieves documents using a single embedding of the user's
# question. This works well when the query phrasing closely matches the
# document text, but often misses relevant chunks whose vocabulary differs.
#
# Multi-Query Retriever solves this by:
#   1. Using an LLM to generate N alternative phrasings of the original query
#   2. Running a vector similarity search for each phrasing independently
#   3. Taking the union of all result sets and deduplicating by content
#
# The final context passed to the answering LLM is richer and more diverse,
# improving recall without any manual prompt engineering from the user.
# =============================================================================

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load OPENAI_API_KEY (and any other env vars) from a local .env file
load_dotenv()

# ── Embedding & LLM setup ────────────────────────────────────────────────────

# text-embedding-3-small converts text into dense vectors; these vectors
# are what the vector store uses to measure semantic similarity at query time
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# This LLM is used ONLY by MultiQueryRetriever to generate alternative query
# phrasings — it is NOT the model that produces the final answer
# temperature=0.3 adds a little creativity so the variants aren't near-identical
llm = ChatOpenAI(model="gpt-5-mini", temperature=0.3)

# ── Sample documents ─────────────────────────────────────────────────────────
# Six domain-specific passages covering biotechnology, cybersecurity,
# neuroscience, renewable energy, robotics, and genetic engineering.
# Real-world pipelines would load these from PDFs, databases, or web crawls.

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
# Long documents are chunked before embedding so that:
#   - Each chunk represents a focused, coherent idea (better embeddings)
#   - Retrieved context fits within the LLM's context window
#
# chunk_size=300   → each chunk is at most 300 characters
# chunk_overlap=50 → adjacent chunks share 50 characters, preventing a
#                    sentence from being cut in half at a boundary
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(docs)
print(f"\nChunks: {len(docs)}\n")

# ── Vector store & base retriever ────────────────────────────────────────────
# InMemoryVectorStore stores chunk embeddings in RAM — no external DB needed.
# from_documents() calls the embedding model once per chunk and indexes the
# resulting vectors so cosine/dot-product similarity can be computed at query time.
vectorstore = InMemoryVectorStore.from_documents(docs, embedding=embeddings)

# base_retriever is a standard similarity-search retriever.
# k=3 means it returns the 3 most similar chunks for any given query.
# MultiQueryRetriever wraps this retriever and calls it multiple times —
# once per generated query variant — so k=3 per variant, not total.
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Baseline: single-query retrieval ─────────────────────────────────────────
# Run the query as-is against the vector store. This shows what a standard
# RAG pipeline would retrieve — useful as a recall baseline to compare against
# the multi-query results below.
query = "How are modern technologies improving human health?"
print("\nQuery:",query)
results_with_sim_search = base_retriever.invoke(query)
print(f"Retrieved {len(results_with_sim_search)} unique documents:\n")

for i, doc in enumerate(results_with_sim_search):
        print(f"--- Result {i+1} [{doc.metadata.get('topic')}] ---")
        print(doc.page_content)
        print()

# ── Multi-Query Retriever ─────────────────────────────────────────────────────
# MultiQueryRetriever generates multiple alternative phrasings of the user question,
# retrieves docs for each, and returns the deduplicated union — expanding recall
# without requiring the user to manually craft multiple queries
#
# Internally it:
#   1. Calls `llm` with a prompt asking for N rephrased versions of `query`
#   2. Parses the LLM output into a list of query strings
#   3. Calls `base_retriever.invoke()` for each variant
#   4. Merges all result lists and removes duplicate chunks
#      (deduplication is by document content hash, not by topic)
#
# Cost note: this makes (N+1) LLM calls total — N for variant generation,
# plus 1 for the final answer — so latency and token usage are higher than
# single-query RAG. Use it when recall matters more than speed.
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
results = retriever.invoke(query)

# Expect more results than the baseline (up to N × k before deduplication).
# Topics that share semantic overlap with any query variant will surface here
# even if they ranked low under the original phrasing.
print(f" Multi-Retrieved {len(results)} unique documents:\n")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} [{doc.metadata.get('topic')}] ---")
    print(doc.page_content)
    print()