import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# Load OPENAI_API_KEY from .env
load_dotenv()


# 12 documents spanning health, programming, history, and nature
# Docs 1-2: contain the exact word "vaccine" — BM25 keyword match
# Docs 3-5: semantically related (immune system, antibodies, herd immunity) but lack the word "vaccine"
#            — dense search finds these, BM25 misses them
# Docs 6-12: off-topic
docs = [
    Document(page_content="Vaccines work by introducing a weakened or inactivated pathogen to trigger an immune response.", metadata={"topic": "health"}),
    Document(page_content="The flu vaccine is reformulated each year to match the most prevalent circulating virus strains.", metadata={"topic": "health"}),
    Document(page_content="The immune system produces antibodies that recognise and neutralise foreign pathogens in the body.", metadata={"topic": "health"}),
    Document(page_content="Herd immunity occurs when enough of a population becomes resistant to a disease, slowing its spread.", metadata={"topic": "health"}),
    Document(page_content="White blood cells called B-lymphocytes produce proteins that bind to and destroy specific antigens.", metadata={"topic": "health"}),
    Document(page_content="Version control systems like Git track changes to code and enable collaboration across teams.", metadata={"topic": "programming"}),
    Document(page_content="Docker containers package applications with their dependencies for consistent deployment.", metadata={"topic": "programming"}),
    Document(page_content="The French Revolution began in 1789 and fundamentally transformed European political structures.", metadata={"topic": "history"}),
    Document(page_content="The Silk Road was an ancient trade network connecting China to the Mediterranean world.", metadata={"topic": "history"}),
    Document(page_content="The Amazon rainforest produces about 20% of the world's oxygen and houses 10% of all species.", metadata={"topic": "nature"}),
    Document(page_content="Coral reefs cover less than 1% of the ocean floor but support about 25% of all marine species.", metadata={"topic": "nature"}),
    Document(page_content="REST APIs communicate over HTTP using standard methods like GET, POST, PUT, and DELETE.", metadata={"topic": "programming"}),
]

# Dense retriever: ChromaDB with OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="hybrid_search",
)

#similarity search
chroma_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

# Sparse retriever: BM25Plus variant on raw text, no embeddings
# BM25Plus ensures every matched term contributes a positive score,
# which improves recall for short documents like the ones we have here
bm25_retriever = BM25Retriever.from_documents(
    docs,
    k=2,
    bm25_variant="plus"
)

print("\n hybraid vector store ready..")

# EnsembleRetriever merges results from both retrievers using Reciprocal Rank Fusion (RRF)
ensemble_retriever = EnsembleRetriever(
    retrievers=[chroma_retriever, bm25_retriever],
    weights=[0.8, 0.2]
)

query = "How do vaccines work to protect against diseases?"
print("\nQuery:", query)
print()

bm25_results = bm25_retriever.invoke(query)
chroma_results = chroma_retriever.invoke(query)
ensemble_results = ensemble_retriever.invoke(query)

# BM25 matches on the exact word "vaccine" — finds docs 1 and 2
# but misses the semantically related immune/antibody docs (3, 4, 5)
print("=== BM25 Only (keyword match) ===")
for i, doc in enumerate(bm25_results, 1):
    print(f"  [{i}] topic={doc.metadata['topic']}: {doc.page_content}")

print()

# Dense search finds docs 3, 4, 5 through semantic understanding
# even though they don't contain the word "vaccine"
print("=== ChromaDB Only (semantic match) ===")
for i, doc in enumerate(chroma_results, 1):
    print(f"  [{i}] topic={doc.metadata['topic']}: {doc.page_content}")

print()

# Hybraid Search
print("=== Ensemble / Hybrid (keyword + semantic) ===")
for i, doc in enumerate(ensemble_results, 1):
    print(f"  [{i}] topic={doc.metadata['topic']}: {doc.page_content}")

