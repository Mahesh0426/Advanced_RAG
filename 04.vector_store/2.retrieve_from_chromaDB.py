import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# ── Environment ───────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please add your OPENAI_API_KEY to the .env file.")

# ── Config — must match load_to_chroma.py ────────────────────────────────────
COLLECTION_NAME   = "demo_1"
PERSIST_DIRECTORY = project_root / "chromaDB_langchain"

if not PERSIST_DIRECTORY.exists():
    raise FileNotFoundError(
        f"\nNo Chroma store found at '{PERSIST_DIRECTORY}'. "
        "Run load_to_chroma.py first."
    )

# ── Connect to the existing Chroma store (read-only, no wipe) ────────────────
embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(PERSIST_DIRECTORY),
)

# ── Helper ────────────────────────────────────────────────────────────────────
def print_documents(title: str, docs: list) -> None:
    print(title)
    for i, doc in enumerate(docs, start=1):
        print(f"  {i}. id={doc.id}")
        print(f"     topic={doc.metadata.get('topic')} | doc_number={doc.metadata.get('doc_number')}")
        print(f"     content={doc.page_content}")
    print()

def preview_text(text, limit=80):
    """Return a short preview for cleaner notebook output."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
#──────────────────────────────────────────────────────────────────────────────────────────


# ── 1. GET THE RAW DOCS FROM chromaDB ─────────────────────────────────────────────
raw_records = vector_store.get(include=["metadatas", "documents"])
# print("\nraw_records:",raw_records)
print(f"Total records in collection: {len(raw_records['ids'])}\n")

# ── 2. Fetch specific documents by id ────────────────────────────────────────
last_three_ids = raw_records["ids"][-3:]
selected_docs  = vector_store.get_by_ids(last_three_ids)
print_documents("Last 3 documents (get_by_ids):", selected_docs)

# ── 3. Similarity search ──────────────────────────────────────────────────────
query = "How does RAG help an LLM answer questions using outside knowledge?"

results = vector_store.similarity_search(query, k=3)
print(f"Query: {query}\n")
print_documents("Top-3 similarity results:\n", results)

# ── 4. Similarity search with relevance score ─────────────────────────────────
results_with_score = vector_store.similarity_search_with_score(query=query, k=4)
print(f"\nTop-4 results with score (lower = more similar):")
for i, (doc, score) in enumerate(results_with_score, start=1):
    print(f"  {i}. score={score:.4f} | doc_num={doc.metadata.get('doc_number')} | topic={doc.metadata.get('topic')} | {doc.page_content[:70]}...\n")
print()

# ---------5.UPDATE EXISTING DOCUMNETS-----------------
# We will update one RAG document and one LLM document.
ids_to_update = [raw_records["ids"][3], raw_records["ids"][7]]
print("\nIDs to update:", ids_to_update)

# Build updated Document objects
updated_documents = [
    Document(
        id=ids_to_update[0],
        page_content="UPDATED: RAG pipelines retrieve relevant chunks using vector similarity before generation.",
        metadata={"topic": "RAG", "doc_number": 4},
    ),
    Document(
        id=ids_to_update[1],
        page_content="UPDATED: Clear prompts help LLMs follow instructions and return more useful answers.",
        metadata={"topic": "LLM", "doc_number": 8},
    ),
]

# -------5a. UPDATE — Chroma update by id, so existing docs are replaced -----------
vector_store.update_documents(ids=ids_to_update, documents=updated_documents)
print("\nDocuments updated.")

# Verify
updated = vector_store.get_by_ids(ids_to_update)
print_documents("\nUpdated documents:", updated)

# Read the updated records back from Chroma to confirm the new values were stored.
updated_raw_records = vector_store.get(ids=ids_to_update)

print("Raw records returned by get(ids=ids_to_update):\n")
for doc_id, document_text, metadata in zip(
    updated_raw_records["ids"],
    updated_raw_records["documents"],
    updated_raw_records["metadatas"],
):
    print(f"id={doc_id}")
    print(f"metadata={metadata}")
    print(f"content={preview_text(document_text)}")
    print()


# ---------7. Update Query and retrieve similarity 
updated_query = "How can retrieved context improve an LLM response in RAG?"

results_after_update = vector_store.similarity_search(updated_query, k=3)
print(f"Query: {updated_query}\n")
print_documents("Top-3 similarity results after UPDATE:\n", results_after_update)

#---------8 DELETE THE DOCUMNETS-------------
# Delete the two cricket examples so the final collection is smaller.
ids_to_delete = [raw_records["ids"][8], raw_records["ids"][9]]
print("ids_to_delete: \n", ids_to_delete)

# delete  from chromaDB
vector_store.delete(ids=ids_to_delete)
print("Deleted these ids:")
for doc_id in ids_to_delete:
    print(doc_id)

#-----9. GET THE REMAINING DOCS------------------
remaining_records = vector_store.get()
remaining_ids = remaining_records["ids"]

print(f"Remaining document count: {len(remaining_ids)}\n")
print("Remaining ids:\n")
for doc_id in remaining_ids:
    print(doc_id)

print("\nDeleted ids still present?\n")
for doc_id in ids_to_delete:
    print(f"{doc_id}: {doc_id in remaining_ids}")

# AFTER — uses metadata already fetched from Chroma
print("\nRemaining topics:")
print([meta["topic"] for meta in remaining_records["metadatas"]])


# fetch all remaining one 
stored_records = vector_store.get(include=["embeddings", "metadatas", "documents"])
stored_records.keys()
print_stored_documents(stored_records)