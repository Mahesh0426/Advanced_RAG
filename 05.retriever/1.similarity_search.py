import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load OPENAI_API_KEY from .env
load_dotenv()

# Create a list of documents with metadata
docs = [
    Document(page_content="Rockets work by expelling gas at high speed, generating thrust through Newton's third law of motion.",
             metadata={"topic": "space"}),
    Document(page_content="The International Space Station orbits Earth at about 400 km altitude and travels at 28,000 km/h.",
             metadata={"topic": "space"}),
    Document(page_content="Spacecraft use gravitational slingshots around planets to gain speed without burning extra fuel.",
             metadata={"topic": "space"}),
    Document(page_content="NASA's Voyager 1 is the farthest human-made object, now over 23 billion km from the Sun.",
             metadata={"topic": "space"}),
    Document(page_content="Solar sails use radiation pressure from sunlight to slowly propel spacecraft without fuel.",
             metadata={"topic": "space"}),
    Document(page_content="DNA is a double-helix molecule that carries the genetic instructions for all living organisms.",
             metadata={"topic": "biology"}),
    Document(page_content="Photosynthesis allows plants to convert sunlight, water, and CO2 into glucose and oxygen.",
             metadata={"topic": "biology"}),
    Document(page_content="The Roman Empire at its peak covered over 5 million square kilometers across three continents.",
             metadata={"topic": "history"}),
    Document(page_content="The printing press, invented by Gutenberg around 1440, revolutionised the spread of knowledge.",
             metadata={"topic": "history"}),
    Document(page_content="The Amazon River discharges more freshwater into the ocean than any other river on Earth.",
             metadata={"topic": "geography"}),
    Document(page_content="The Sahara Desert spans about 9.2 million square kilometers across northern Africa.",
             metadata={"topic": "geography"}),
    Document(page_content="Mitochondria generate ATP through cellular respiration, powering nearly all cellular processes.",
             metadata={"topic": "biology"}),
]

# for index, doc in enumerate(docs,1):
#     print(f"Doc No.: {index} | Topic: {doc.metadata["topic"]}")
#     print(f"Content: {doc.page_content}\n")


# Embed documents and store in an in-memory ChromaDB collection
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="similarity_search_demo",
)
print("\nVector store is ready.")
print("\nlenghth of vectorstore:", len(vectorstore.get()["ids"]))

# Similarity search retriever returns the k most semantically similar documents
# It ranks by cosine similarity between the query embedding and document embeddings
retriever = vectorstore.as_retriever(
    search_type="similarity",  # similarity, similarity_score_threshold, mmr
    search_kwargs={"k": 3},   # k is the number of retrieved docs
)

print("\nretriever:", retriever)

# WITHOUT retriever
# this method is not used generally inside chains, or if you require customizations then 
# also you do not use this method

query = "How do cells generate their energy?"
results = vectorstore.similarity_search(query, k=3)

print(f"{query}\n")
for i, doc in enumerate(results, 1):
    print(f"Result {i} [topic={doc.metadata['topic']}]")
    print(f"{doc.page_content}")
    print()


# retrievers in langchain are runnables, create chains using retrievers

results2 = retriever.invoke(query)

# All returned documents should be from the space topic
print(f"{query}\n")
for i, doc in enumerate(results2, 1):
    print(f"Result {i} [topic={doc.metadata['topic']}]")
    print(f"{doc.page_content}")
    print()