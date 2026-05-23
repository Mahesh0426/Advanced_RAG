import os
import shutil
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings



# go one level up from the current file → that's the project root
project_root = Path(__file__).parent.parent

dotenv_path = project_root / ".env"
load_dotenv(dotenv_path=dotenv_path)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please add your OPENAI_API_KEY to the .env file before running this notebook.")

print(f"Loaded environment from: {dotenv_path} \n")

# Use a fixed collection name and persistence path so each rerun is predictable.
collection_name = "demo_2"
persist_directory = project_root / "chromaDB_langchain"

print(f"Collection name: {collection_name}")
print(f"Persist directory: {persist_directory} \n")


# Start fresh so the CRUD flow produces the same result each time.
if persist_directory.exists():
    shutil.rmtree(persist_directory)
    print(" Removed the old Chroma directory.")
else:
    print("No previous Chroma directory was found.\n")


# Create the embedding model and connect it to a persistent Chroma store.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#initialize chroma DB | vector DB
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=str(persist_directory),
)

print("\nVector store is ready.")


#helper function
def preview_text(text, limit=80):
    """Return a short preview for cleaner notebook output."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def print_documents(title, docs):
    """Print Document objects in a beginner-friendly format."""
    print(title)
    for index, doc in enumerate(docs, start=1):
        print(f"{index}. id={doc.id}")
        print(f"   topic={doc.metadata.get('topic')} | doc_number={doc.metadata.get('doc_number')}")
        print(f"   content={doc.page_content}")
    print()

#document example 
# Keep the raw sample data separate from the Document objects so it is easier to read.
document_examples = [
    {
        "topic": "AI",
        "doc_number": 1,
        "text": "Artificial intelligence helps machines perform tasks that usually need human reasoning.",
    },
    {
        "topic": "AI",
        "doc_number": 2,
        "text": "AI systems can analyze patterns in data to support predictions and automation.",
    },
    {
        "topic": "AI",
        "doc_number": 3,
        "text": "Responsible AI development includes fairness, transparency, and safety checks.",
    },
    {
        "topic": "RAG",
        "doc_number": 4,
        "text": "RAG combines retrieval with generation so the model can answer using external knowledge.",
    },
    {
        "topic": "RAG",
        "doc_number": 5,
        "text": "A retriever in a RAG pipeline finds relevant chunks before the language model generates an answer.",
    },
    {
        "topic": "RAG",
        "doc_number": 6,
        "text": "Vector stores are important in RAG because they make semantic search over embedded documents possible.",
    },
    {
        "topic": "LLM",
        "doc_number": 7,
        "text": "LLMs generate text by predicting likely next tokens from patterns learned during training.",
    },
    {
        "topic": "LLM",
        "doc_number": 8,
        "text": "Prompt design can improve how clearly an LLM follows instructions and returns useful answers.",
    },
    {
        "topic": "Cricket",
        "doc_number": 9,
        "text": "Cricket teams score runs through batting partnerships, boundaries, and quick running between the wickets.",
    },
    {
        "topic": "Cricket",
        "doc_number": 10,
        "text": "A cricket bowler can pressure batters with pace, swing, spin, and accurate line and length.",
    },
]

# print(f"Prepared {len(document_examples)} document examples.")

# prepare for 10n doc example
# for doc in document_examples:
#     print(doc)
#     print()


# print(uuid4())

# Convert the sample data into LangChain Document objects.
documents = [
    Document(
        id=str(uuid4()),
        page_content=item["text"],
        metadata={"topic": item["topic"], "doc_number": item["doc_number"]},
    )
    for item in document_examples
]

# print_documents("Dummy documents prepared:", documents)

print("First document:",documents[0].id, "\n\n")

#INSERT TO CHROMADB
# Insert the documents into Chroma. Chroma creates embeddings during this step.
document_ids = vector_store.add_documents(documents)

print("Inserted document ids:")
for doc_id in document_ids:
    print(doc_id)

print(f"\nTotal inserted documents: {len(document_ids)}")

# Read the store data back 
# The get() method returns the low-level Chroma record structure.
raw_records = vector_store.get(include=["embeddings", "metadatas", "documents"])
print("\nkeys:",raw_records.keys())
# print("raw_records: ", raw_records)


# print(raw_records["embeddings"][0:2, 0:20])
# print(raw_records["embeddings"][0:2, 0:20]).shape

# print(f"\nTotal records in collection: {len(raw_records['ids'])}")
# print("First three ids from get():")
# for doc_id in raw_records["ids"][:3]:
#     print(doc_id)


# Pick a few ids so we can read them back in a higher-level format.
selected_ids = document_ids[-3:]
selected_ids

# get_by_ids() returns LangChain Document objects instead of the raw Chroma dictionary.
selected_documents = vector_store.get_by_ids(selected_ids)
print_documents("\nDocuments fetched with get_by_ids():", selected_documents)

query = "How does RAG help an LLM answer questions using outside knowledge?"

search_results = vector_store.similarity_search(query, k=3)
print(f"\nQuery: {query}\n")
print("Similarity search results:", search_results)

# print("\nsearch_results:", search_results)