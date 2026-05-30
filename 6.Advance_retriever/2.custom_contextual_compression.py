# =============================================================================
# CUSTOM CONTEXTUAL COMPRESSION RETRIEVER
# =============================================================================
# This module re-implements LangChain's built-in contextual compression stack
# from scratch to make the internals fully visible and customisable.
#
# Two custom classes are built:
#
#   CustomLLMExtractorChain
#     └── mirrors LangChain's LLMChainExtractor
#         Takes retrieved docs + a query, calls an LLM on each doc, and
#         returns only the sentences relevant to the query.
#         Docs where nothing is relevant are dropped entirely (NO_OUTPUT).
#
#   CustomContextualCompressionRetriever
#     └── mirrors LangChain's ContextualCompressionRetriever
#         Extends BaseRetriever to compose the base vector-store retriever
#         with the extractor above into a single .invoke()-able interface.
#
# Why build custom instead of using LangChain's built-in?
#   - Swap or modify the extraction prompt (e.g. summarise vs extract verbatim)
#   - Add per-doc scoring, logging, or confidence thresholds
#   - Run compression in parallel with asyncio instead of serially
#   - Full visibility into every step for debugging and experimentation
#
# Data flow:
#   query
#     → CustomContextualCompressionRetriever._get_relevant_documents()
#         → base_retriever.invoke(query)            [vector similarity, top-k]
#         → base_compressor.compress_documents()    [LLM trims each doc]
#             → prompt | llm | StrOutputParser()    [LCEL chain, one call/doc]
#     → list[Document] — trimmed page_content, original metadata intact
# =============================================================================

from dotenv import load_dotenv
from typing import Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# Load OPENAI_API_KEY (and any other vars) from the .env file.
# Must be called before instantiating any OpenAI client.
load_dotenv()

# -----------------------------------------------------------------------------
# MODELS
# -----------------------------------------------------------------------------

# Embedding model: converts text → 1536-dim dense vector.
# Used at index time (docs) and at query time (query) — must be the same model
# for cosine similarity to be meaningful across both.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chat model used inside CustomLLMExtractorChain to read each retrieved doc
# and extract only the sentences relevant to the query.
# temperature=0 → deterministic output; no creative variation wanted here.
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

# -----------------------------------------------------------------------------
# DOCUMENTS
# -----------------------------------------------------------------------------
# Six manually created Document objects, each covering a different topic.
# Each doc deliberately mixes on-topic and off-topic sentences so the
# LLM extractor has meaningful noise to identify and strip out.
#
# Document anatomy:
#   page_content → the raw text that gets embedded and searched
#   metadata     → arbitrary key-value pairs (topic, source, page number, etc.)
#                  that survive both retrieval and compression unchanged —
#                  important for citations and downstream filtering
#
# In production these would come from document loaders:
#   PyPDFLoader, WebBaseLoader, CSVLoader, WikipediaLoader, JSONLoader, etc.

docs = [
    Document(
        page_content=(
            "Artificial intelligence has made remarkable strides in natural language processing, "
            "with large language models now capable of generating human-quality text and code. "
            "Computer vision systems can identify objects in images with superhuman accuracy, "
            "powering applications from autonomous vehicles to medical imaging diagnostics. "
            "However, the rapid advancement of AI has raised significant ethical concerns about "
            "job displacement, algorithmic bias, and the concentration of power among a few tech companies."
        ),
        metadata={"topic": "artificial_intelligence"},
    ),
    Document(
        page_content=(
            "Global temperatures have risen by approximately 1.1 degrees Celsius since pre-industrial "
            "times, driven primarily by the burning of fossil fuels. The melting of polar ice caps has "
            "accelerated, contributing to rising sea levels that threaten coastal communities worldwide. "
            "Renewable energy adoption is growing rapidly, with solar and wind power becoming cheaper "
            "than coal in many regions. Governments are implementing carbon pricing mechanisms and "
            "investing in green infrastructure to meet Paris Agreement targets."
        ),
        metadata={"topic": "climate_change"},
    ),
    Document(
        page_content=(
            "NASA's Artemis program aims to return humans to the Moon by the mid-2020s, establishing "
            "a sustainable presence as a stepping stone to Mars. Private companies like SpaceX are "
            "developing reusable rocket technology that has dramatically reduced launch costs. "
            "The James Webb Space Telescope has captured unprecedented images of distant galaxies, "
            "revealing new insights about the early universe. Asteroid mining is being explored as a "
            "potential source of rare minerals needed for electronics manufacturing."
        ),
        metadata={"topic": "space_exploration"},
    ),
    Document(
        page_content=(
            "CRISPR gene editing technology has revolutionized medical genomics, enabling precise "
            "modifications to DNA sequences that were previously impossible. Researchers are using "
            "genomic data to develop personalized medicine approaches, tailoring treatments based on "
            "an individual's genetic profile. Recent breakthroughs in mRNA technology, accelerated by "
            "COVID-19 vaccine development, are now being applied to cancer immunotherapy and rare "
            "genetic disorders. Hospital information systems are increasingly integrating genomic data "
            "to support clinical decision-making at the point of care."
        ),
        metadata={"topic": "medicine"},
    ),
    Document(
        page_content=(
            "The global economy is navigating a period of high inflation driven by supply chain "
            "disruptions, energy price volatility, and post-pandemic demand surges. Central banks "
            "worldwide have raised interest rates aggressively to combat inflation, impacting housing "
            "markets and consumer spending. Cryptocurrency regulation is becoming a priority for "
            "financial authorities, with the EU's MiCA framework setting a global precedent. "
            "Trade tensions between major economies continue to reshape global supply chains, "
            "pushing companies toward nearshoring and diversification strategies."
        ),
        metadata={"topic": "economics"},
    ),
    Document(
        page_content=(
            "Quantum computing has reached a critical milestone with several companies demonstrating "
            "quantum advantage on specific computational tasks. Error correction remains the biggest "
            "challenge, as current quantum processors are highly susceptible to noise and decoherence. "
            "Quantum simulation of molecular structures could transform drug discovery by accurately "
            "modeling protein folding and chemical interactions. Major tech companies and governments "
            "are investing billions in quantum research, viewing it as essential for national security "
            "and economic competitiveness."
        ),
        metadata={"topic": "quantum_computing"},
    ),
]

print(f"Total documents: {len(docs)}")

# -----------------------------------------------------------------------------
# VECTOR STORE & BASE RETRIEVER
# -----------------------------------------------------------------------------

# Embeds all 6 docs on creation and stores (vector, Document) pairs in RAM.
# InMemoryVectorStore is suitable for prototyping and small corpora.
# For production swap with: Chroma, Pinecone, Weaviate, pgvector, FAISS, etc.
vectorstore = InMemoryVectorStore.from_documents(docs, embedding=embeddings)

# Wrap the vector store as a LangChain Retriever interface.
# k=3 → always fetch the 3 docs with highest cosine similarity to the query.
# These 3 full docs are the raw candidates passed into the compression layer.
# Increasing k gives more candidates but raises LLM compression cost linearly.
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# =============================================================================
# CLASS 1 — CustomLLMExtractorChain
# =============================================================================
# Mirrors LangChain's built-in LLMChainExtractor.
# Responsible for reading each retrieved doc and extracting only the portion
# that is directly relevant to the query, using an LLM chain internally.

# Custom LLM extractor that mirrors LLMChainExtractor using a Runnable chain internally
class CustomLLMExtractorChain:
    """Extracts relevant portions from documents using an LLM chain."""

    def __init__(self, llm_chain):
        # Stores the compiled LCEL chain built by from_llm().
        # Not meant to be called directly — use from_llm() as the entry point.
        self.llm_chain = llm_chain

    @classmethod
    def from_llm(cls, llm):
        # Same prompt pattern used by LangChain's LLMChainExtractor

        # Extraction prompt — two critical constraints given to the LLM:
        #   1. Extract *AS IS* — no paraphrasing, no summarising; verbatim only.
        #      This ensures the output is grounded in the source text exactly.
        #   2. Return the literal string "NO_OUTPUT" when nothing is relevant.
        #      This sentinel lets compress_documents() detect and silently drop
        #      the doc without having to parse ambiguous empty strings.
        prompt = ChatPromptTemplate.from_template(
            "Given the following question and context, extract any part of the context "
            "*AS IS* that is relevant to answer the question. If none of the context is "
            "relevant return NO_OUTPUT.\n\n"
            "Remember, *DO NOT* edit the extracted parts of the context.\n\n"
            "> Question: {question}\n"
            "> Context:\n>>>\n{context}\n>>>\n"
            "Extracted relevant parts:"
        )
        # LCEL chain — the | operator pipes output of each step to the next:
        #   prompt          → formats {question} + {context} into a ChatMessage
        #   llm             → sends the message to the model, returns AIMessage
        #   StrOutputParser → extracts the plain string from AIMessage.content

        # create the chain
        llm_chain = prompt | llm | StrOutputParser()

        # cls(...) calls __init__ with the compiled chain — factory pattern
        # keeps chain construction logic separate from instance storage
        return cls(llm_chain=llm_chain)

    def compress_documents(self, documents: list[Document], query: str) -> list[Document]:
        # Accumulates docs that passed LLM extraction (i.e. not NO_OUTPUT)
        compressed = []
        for doc in documents:
            # Invoke the LCEL chain — sends {question: query, context: doc text}
            # to the LLM and returns the extracted relevant text as a plain string
            result = self.llm_chain.invoke({"question": query, "context": doc.page_content})
            result = result.strip()
            # Drop the doc if the LLM found nothing relevant in it.
            # Checking for both empty string and "NO_OUTPUT" guards against
            # minor LLM formatting variations (e.g. trailing whitespace).
            if result and result != "NO_OUTPUT":
                # Rebuild as a new Document:
                #   page_content → replaced with the trimmed, relevant text only
                #   metadata     → copied from original so topic/source is preserved
                compressed.append(Document(page_content=result, metadata=doc.metadata))
        return compressed


# =============================================================================
# CLASS 2 — CustomContextualCompressionRetriever
# =============================================================================
# Mirrors LangChain's built-in ContextualCompressionRetriever.
# Composes the base vector-store retriever with CustomLLMExtractorChain into
# a single object that exposes the standard LangChain Retriever interface.
#
# Extending BaseRetriever (rather than being a plain class) means:
#   - .invoke(), .stream(), .batch() and async variants all work automatically
#   - Works inside any LangChain chain or agent without extra wiring
#   - Callbacks and tracing (LangSmith) are supported out of the box

# Custom retriever that composes base retrieval with LLM compression
class CustomContextualCompressionRetriever(BaseRetriever):
    """Retriever that compresses documents using a custom LLM extractor chain."""

    # Pydantic-style field declarations required by BaseRetriever.
    # LangChain uses Pydantic v1 under the hood — fields must be declared at
    # class level so the model schema is built correctly before instantiation.
    base_retriever: BaseRetriever
    base_compressor: Any  # CustomLLMExtractorChain instance
    # typed as Any so non-LangChain compressors (plain Python classes) are accepted

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # The single method BaseRetriever requires you to implement.
        # LangChain routes all public calls (.invoke, .stream, async) through
        # here — implementing just this one method gives full interface support.

        # Step 1: retrieve documents from the base retriever
        # Fetches the top-k docs from the vector store via cosine similarity.
        # Returns full, unmodified Documents — no compression applied yet.
        docs = self.base_retriever.invoke(query)

        # Step 2: compress using the LLM extractor
        # Passes each fetched doc + the query to CustomLLMExtractorChain.
        # Returns only the docs (and trimmed content) the LLM found relevant.
        compressed_docs = self.base_compressor.compress_documents(docs, query)
        return compressed_docs


# =============================================================================
# WIRING & EXECUTION
# =============================================================================

# Wire up the custom classes and run the same query as notebook 1

# Build the extractor — constructs and compiles the prompt | llm | parser chain
compressor = CustomLLMExtractorChain.from_llm(llm)

# Compose base retriever + compressor into a single retriever interface.
# Calling .invoke(query) on custom_retriever internally runs:
#   fetch (base_retriever) → compress (base_compressor) → return results
custom_retriever = CustomContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor,
)

query = "How is CRISPR acting as a big enabler in creating personalized medicine?"

# .invoke() calls _get_relevant_documents() internally via BaseRetriever
results = custom_retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"--- Result {i+1} [{doc.metadata.get('topic')}] ---")
    print(doc.page_content)
    print()