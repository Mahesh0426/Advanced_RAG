import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4


# Load OPENAI_API_KEY from .env
load_dotenv()

# 12 documents: ML (highly relevant to query), economics (somewhat relevant), cooking and sports (off-topic)
# The threshold filter will progressively cut out the lower-scoring documents
docs = [
    Document(page_content="Supervised learning trains models on labeled input-output pairs to predict unseen data.", metadata={"topic": "ml"}),
    Document(page_content="Neural networks learn by adjusting weights through backpropagation and gradient descent.", metadata={"topic": "ml"}),
    Document(page_content="Overfitting occurs when a model memorises training data and performs poorly on new data.", metadata={"topic": "ml"}),
    Document(page_content="Training data quality and quantity are the most important factors in model performance.", metadata={"topic": "ml"}),
    Document(page_content="Cross-validation splits data into folds to evaluate model generalisation more reliably.", metadata={"topic": "ml"}),
    Document(page_content="Inflation is the rate at which the general level of prices for goods and services rises over time.", metadata={"topic": "economics"}),
    Document(page_content="GDP measures the total monetary value of all goods and services produced within a country.", metadata={"topic": "economics"}),
    Document(page_content="Interest rates set by central banks influence borrowing costs and consumer spending.", metadata={"topic": "economics"}),
    Document(page_content="Caramelisation occurs when sugar is heated above 160°C, creating complex flavour compounds.", metadata={"topic": "cooking"}),
    Document(page_content="Fermentation uses microorganisms to convert sugars into alcohol or acids, preserving food.", metadata={"topic": "cooking"}),
    Document(page_content="A marathon is a long-distance race of exactly 42.195 kilometres, run on roads.", metadata={"topic": "sports"}),
    Document(page_content="Tennis scoring follows a love, 15, 30, 40, game sequence, with deuce at 40-40.", metadata={"topic": "sports"}),
]

# Embed documents and store in ChromaDB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name="demo",
    collection_configuration={
        "hnsw": {
            "space": "cosine"      # cosine, L2, ip
        }
    }
)

print("vectorstore:", vectorstore)
print("\nVector store is ready.")


# add documents to vector store
ids_added = vectorstore.add_documents(documents=docs,
                                      ids=[str(uuid4()) for _ in range(len(docs))])

print(f"\nInserted {len(ids_added)} documents into 'demo'.")
print("ids_added: ",ids_added)

# # retrieve top 5 documents with score
query = "How does ML model training work?"

similarity_scores = vectorstore.similarity_search_with_score(query, k=3)

for doc, score in similarity_scores:
    print(f"\nScore: {score:.4f} | Topic: {doc.metadata['topic']} | Content: {doc.page_content}\n")
print("\n")


# get all the scores in (1-score) format
# actual cosine similarity scores
for _, score in similarity_scores:
    print(f"Cosine Similarity Score: {1 - score:.4f}")
print("\n")

# Using similarity_score_threshold method
query = "How does ML model training work?"
threshold = 0.43   

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": threshold},
)

results = retriever.invoke(query)

print(f"\n=== Threshold = {threshold} — {len(results)} document(s) returned ===")
for i, doc in enumerate(results, 1):
    print(f"  [{i}] topic={doc.metadata['topic']}: {doc.page_content}")
