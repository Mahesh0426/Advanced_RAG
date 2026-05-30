# =============================================================================
# CONTEXTUAL COMPRESSION RAG PIPELINE
# =============================================================================
# Standard RAG retrieves full documents even when only a small portion is
# relevant to the query. Contextual Compression solves this by post-processing
# retrieved docs — either dropping irrelevant ones entirely (EmbeddingsFilter)
# or trimming them to only the relevant sentences (LLMChainExtractor).
#
# Pipeline overview:
#   Query → VectorStore → base_retriever (top-k docs)
#                                   ↓
#                         [Compressor layer]
#                         ├── EmbeddingsFilter       (cheap: drop by score)
#                         ├── LLMChainExtractor      (precise: trim by LLM)
#                         └── Pipeline(filter → extract)  (best of both)

#   ┌─────────────────────────────────────────────────────────────────────┐
#   │  Strategy               │ What it does          │ Cost              │
#   ├─────────────────────────┼───────────────────────┼───────────────────┤
#   │  LLMChainExtractor      │ Trims doc to relevant │ 1 LLM call / doc  │
#   │                         │ sentences via LLM     │ (slow, precise)   │
#   ├─────────────────────────┼───────────────────────┼───────────────────┤
#   │  EmbeddingsFilter       │ Drops whole docs below│ No LLM call       │
#   │                         │ cosine sim threshold  │ (fast, coarse)    │
#   ├─────────────────────────┼───────────────────────┼───────────────────┤
#   │  DocumentCompressor     │ Filter first (cheap), │ LLM only on       │
#   │  Pipeline               │ then extract (precise)│ survivors         │
#   └─────────────────────────┴───────────────────────┴───────────────────┘
# =============================================================================

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)

# Load OPENAI_API_KEY (and any other vars) from the .env file
load_dotenv()


# =============================================================================
# MODELS
# =============================================================================

# Embedding model: converts text → dense vector (1536 dims for text-embedding-3-small)
# Used for both indexing documents and encoding the query at retrieval time
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# LLM: used only by LLMChainExtractor to read a doc and extract relevant sentences
# temperature=0 keeps output deterministic (no creativity needed here)
llm = ChatOpenAI(model="gpt-4-mini", temperature=0)


# =============================================================================
# DOCUMENTS
# =============================================================================
# Each Document has:
#   - page_content : the raw text that gets embedded and searched
#   - metadata     : arbitrary key-value pairs preserved through retrieval
#                    (useful for filtering, citations, or display)
#
# In a real pipeline these come from loaders (PyPDFLoader, WebBaseLoader, etc.)
# Here we create them manually to keep the example self-contained.
# Each doc intentionally mixes relevant and tangential info to show how
# compression strips out the noise.

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

print(f"Created {len(docs)} documents\n")


# =============================================================================
# VECTOR STORE & BASE RETRIEVER
# =============================================================================

# InMemoryVectorStore embeds all docs on creation and stores (vector, doc) pairs
# in RAM. Fine for prototyping; swap with Chroma/Pinecone/Weaviate for production.
vectorstore = InMemoryVectorStore.from_documents(docs, embedding=embeddings)

# as_retriever() wraps the vector store as a LangChain Retriever interface.
# search_kwargs={"k": 3} → always return the 3 most similar docs to the query.
# These 3 docs are the raw candidates that get passed to the compressor layer.
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# =============================================================================
# QUERY
# =============================================================================

query = "How is CRISPR acting as a big enabler in creating personalized medicine?"


# =============================================================================
# LAYER 1 — BASELINE (no compression)
# =============================================================================
# Uncomment to see what the raw top-3 docs look like before any compression.
# Useful to understand what the compressors are working with.

# base_results = base_retriever.invoke(query)
# for i, doc in enumerate(base_results):
#     print(f"--- Base Result {i+1} [{doc.metadata.get('topic')}] ---")
#     print(doc.page_content)
#     print()


# =============================================================================
# LAYER 2A — LLMChainExtractor
# =============================================================================
# How it works:
#   1. base_retriever fetches top-3 full docs
#   2. For each doc, sends (query + doc) to the LLM
#   3. LLM returns only the sentences from that doc that are relevant to query
#   4. Doc is kept but its page_content is replaced with the extracted subset
#
# Pros: very precise — returns exact relevant sentences
# Cons: 1 LLM call per retrieved doc → slower and more expensive

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,   # post-processor that trims doc content
    base_retriever=base_retriever,  # upstream retriever that fetches candidates
)

# Uncomment to see LLM-extracted sentences per doc:
# compressed_results = compression_retriever.invoke(query)
# for i, doc in enumerate(compressed_results):
#     print(f"--- Compressed Result {i+1} [{doc.metadata.get('topic')}] ---")
#     print(doc.page_content)
#     print()


# =============================================================================
# LAYER 2B — EmbeddingsFilter
# =============================================================================
# How it works:
#   1. base_retriever fetches top-3 full docs
#   2. Embeds the query and each doc using the same embedding model
#   3. Computes cosine similarity between query vector and each doc vector
#   4. Drops the entire doc if its similarity score < similarity_threshold
#   5. Surviving docs are returned with their full, unmodified content
#
# Pros: no LLM call needed — fast and cheap
# Cons: keeps the whole doc (no sentence-level trimming), all-or-nothing per doc
#
# Tuning similarity_threshold:
#   - Too high (e.g. 0.80) → filters out most docs, empty results
#   - Too low  (e.g. 0.10) → lets everything through, no real filtering
#   - Sweet spot for text-embedding-3-small is typically 0.45–0.60
#
# Debug tip: set threshold=0.0 first to see all scores, then tune upward:
#   score = doc.metadata.get("relevance_score")  ← added by EmbeddingsFilter

embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.50,  # drop docs scoring below this cosine similarity
)

compression_retriever_emb = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever,
)

print("=== EmbeddingsFilter Results ===\n")
emb_results = compression_retriever_emb.invoke(query)
for i, doc in enumerate(emb_results):
    # relevance_score is injected into metadata by EmbeddingsFilter
    # (key name may vary by LangChain version — print doc.metadata to inspect)
    score = doc.metadata.get("relevance_score", "N/A")
    print(f"--- Filtered Result {i+1} [{doc.metadata.get('topic')}] (score: {score}) ---")
    print(doc.page_content)
    print()


# =============================================================================
# LAYER 3 — DocumentCompressorPipeline (EmbeddingsFilter → LLMChainExtractor)
# =============================================================================
# Chains multiple compressors in sequence. Each compressor's output becomes
# the next compressor's input. This is the recommended production pattern:
#
#   Step 1 — EmbeddingsFilter (cheap gate):
#     Drops docs below the similarity threshold with no LLM cost.
#     e.g. of 3 retrieved docs, 2 survive the filter.
#
#   Step 2 — LLMChainExtractor (precise trimmer):
#     Only runs on the docs that survived step 1.
#     Trims each survivor to its relevant sentences.
#
# Net effect: LLM calls are minimised because irrelevant docs are filtered
# before the expensive extraction step runs.

pipeline_compressor = DocumentCompressorPipeline(
    transformers=[embeddings_filter, compressor]  # order matters: filter first, extract second
)

compression_retriever_pipeline = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever,
)

print("=== Pipeline Results (EmbeddingsFilter → LLMChainExtractor) ===\n")
pipeline_results = compression_retriever_pipeline.invoke(query)
for i, doc in enumerate(pipeline_results):
    print(f"--- Pipeline Result {i+1} [{doc.metadata.get('topic')}] ---")
    print(doc.page_content)
    print()