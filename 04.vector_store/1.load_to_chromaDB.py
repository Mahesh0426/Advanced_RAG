import os
import shutil
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# ── Environment ──────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please add your OPENAI_API_KEY to the .env file.")

# ── Config ───────────────────────────────────────────────────────────────────
COLLECTION_NAME   = "demo_1"
PERSIST_DIRECTORY = project_root / "chromaDB_langchain"

# ── Fresh start (wipe old data so each run is predictable) ───────────────────
if PERSIST_DIRECTORY.exists():
    shutil.rmtree(PERSIST_DIRECTORY)
    print("Removed old Chroma directory.")
else:
    print("No previous Chroma directory found.")

# ── Raw sample data ───────────────────────────────────────────────────────────
document_examples = [
    {"topic": "AI",      "doc_number": 1,  "text": "Artificial intelligence helps machines perform tasks that usually need human reasoning."},
    {"topic": "AI",      "doc_number": 2,  "text": "AI systems can analyze patterns in data to support predictions and automation."},
    {"topic": "AI",      "doc_number": 3,  "text": "Responsible AI development includes fairness, transparency, and safety checks."},
    {"topic": "RAG",     "doc_number": 4,  "text": "RAG combines retrieval with generation so the model can answer using external knowledge."},
    {"topic": "RAG",     "doc_number": 5,  "text": "A retriever in a RAG pipeline finds relevant chunks before the language model generates an answer."},
    {"topic": "RAG",     "doc_number": 6,  "text": "Vector stores are important in RAG because they make semantic search over embedded documents possible."},
    {"topic": "LLM",     "doc_number": 7,  "text": "LLMs generate text by predicting likely next tokens from patterns learned during training."},
    {"topic": "LLM",     "doc_number": 8,  "text": "Prompt design can improve how clearly an LLM follows instructions and returns useful answers."},
    {"topic": "Cricket", "doc_number": 9,  "text": "Cricket teams score runs through batting partnerships, boundaries, and quick running between the wickets."},
    {"topic": "Cricket", "doc_number": 10, "text": "A cricket bowler can pressure batters with pace, swing, spin, and accurate line and length."},
]

# ── Build LangChain Document objects ─────────────────────────────────────────
documents = [
    Document(
        id=str(uuid4()),
        page_content=item["text"],
        metadata={"topic": item["topic"], "doc_number": item["doc_number"]},
    )
    for item in document_examples
]
# print ("Documents:", documents)

# ── Connect to Chroma and insert ──────────────────────────────────────────────
embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(PERSIST_DIRECTORY),
)

document_ids = vector_store.add_documents(documents)


print(f"\nInserted {len(document_ids)} documents into '{COLLECTION_NAME}'.\n")
print(f"\nPersisted at: {PERSIST_DIRECTORY}")
for doc_id in document_ids:
    print(" ", doc_id)

print("\nVector store is ready.")