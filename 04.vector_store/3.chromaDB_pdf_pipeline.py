import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Environment ───────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please add your OPENAI_API_KEY to the .env file.")
print(f"\nLoaded environment variables from '{project_root / '.env'}'.")

# ── create the path for pdf  ───────────────────────────────────────────────────────────────
current_dir = Path(__file__).resolve().parent
pdf_path = current_dir.parent  /"04.vector_store"/ "documents" / "beyond-chatbots-ai-agents-next-real-shift.pdf"
# print(pdf_path.exists())
# print("Checking:", pdf_path)



# ── display header helper function ──────────────────
def preview_text(text, limit=120):
    """Return a short preview for cleaner notebook output."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."

def print_documents(title, docs):
    """Print retrieved documents using page metadata and a text preview."""
    print(title)
    for index, doc in enumerate(docs, start=1):
        print(f"{index}. page={doc.metadata.get('page')} | source={doc.metadata.get('source')}")
        print(f"   content={doc.page_content}")
    print()

# ── 1.load the pdf ──────────────────
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()
print(f"\nTotal pages loaded: {len(docs)}")
# print(docs[0].page_content)

# print(f"\nFirst page preview: {preview_text(docs[0].page_content)}")
# print(f"\nFirst page metadata: {docs[0].metadata}")

# ── 2. split pdf into chunk | chunking ──────────────────
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)
print(f"\nTotal chunks created: {len(split_docs)}")
print(f"\nFirst chunk preview: {preview_text(split_docs[0].page_content)}")
print(f"\nFirst chunk metadata: {split_docs[0].metadata}")


# for i, chunk in enumerate(split_docs, 1):
#     print(f"\nChunk {i} ({len(chunk.page_content)} chars): {repr(chunk.page_content)}")

# ── 3. Initialize embedding models ──────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("\nEmbedding model is ready.")

# ── Config ───────────────────────────────────────────────────────────────────
COLLECTION_NAME   = "rag-pipeline"
PERSIST_DIRECTORY = project_root / "chromaDB_langchain"

# ── 4. Initialize chromaDB to store ──────────────────
vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=str(PERSIST_DIRECTORY),
)

print(f"Stored {len(split_docs)} chunks in the '{COLLECTION_NAME}' collection.")

# ── 5. Query────────────────────────────────────
query = "How do AI agents use tools and memory?"
results = vector_store.similarity_search(query, k=3)


print(f"\nQuery: {query}\n")
print_documents("\nRetrieved chunks:\n", results)


retrieved_docs = vector_store.similarity_search_with_score(query, k=3)

for doc, score in retrieved_docs:
    print(f"Score: {score:.4f}")
    print(f"Content preview: {doc.page_content}")
    print(f"page_no. {doc.metadata.get("page_label")}")
    print()