from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

# Dummy documents covering different topics, each with a mix of relevant and tangential info
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

print(f"\nCreated {len(docs)} documents")
print()

# print(docs[0].page_content)

# Build a vector store and base retriever that returns top 3 docs
vectorstore = InMemoryVectorStore.from_documents(docs, embedding=embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Baseline: retrieve without any compression to see full document content
query = "\nHow is CRISPR acting as a big enabler in creating personalized medicine?\n"
print()

# base_results = base_retriever.invoke(query)
# for i, doc in enumerate(base_results):
#     print(f"--- Result {i+1} [{doc.metadata.get('topic')}] ---")
#     print(doc.page_content)
#     print()


# LLMChainExtractor: uses an LLM to extract only the relevant portions from each document
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# compressed_results = compression_retriever.invoke(query)
# for i, doc in enumerate(compressed_results):
#     print(f"--- Compressed Result {i+1} [{doc.metadata.get('topic')}] ---")
#     print(doc.page_content)
#     print()


# EmbeddingsFilter: drops entire documents whose embedding similarity to the query is below the threshold
#similarity search --> query + retrieved docs from base_retriever --> embedding model
# simaliraty score --> threshold --> filter --> those who have scores greater than the threshold
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.50,
)

compression_retriever_emb = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever,
)

emb_results = compression_retriever_emb.invoke(query)
for i, doc in enumerate(emb_results):
    score = doc.metadata.get("relevance_score", "N/A")
    print(f"--- Filtered Result {i+1} [{doc.metadata.get('topic')}] (score: {score}) ---")
    print(doc.page_content)
    print()


# DocumentCompressorPipeline: chain multiple compressors together
# First filter by embeddings (cheap), then extract with LLM (expensive) — minimizes LLM calls
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[embeddings_filter, compressor]
)

compression_retriever_pipeline = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever,
)

pipeline_results = compression_retriever_pipeline.invoke(query)
for i, doc in enumerate(pipeline_results):
    print(f"--- Pipeline Result {i+1} [{doc.metadata.get('topic')}] ---")
    print(doc.page_content)
    print()