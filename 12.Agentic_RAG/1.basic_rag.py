from langgraph.graph import StateGraph,END,START,MessagesState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

from pathlib import Path

load_dotenv()

base_dir = Path(__file__).resolve().parent
documents_dir = base_dir / "documents"
pdf_path = documents_dir / "evs_oil_price_shock.pdf"

# 1. Load the PDF
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()
print(f"Loaded {len(docs)} pages from the PDF.")
# print(docs[0].page_content)

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"\nSplit into {len(chunks)} chunks.")

# 3. Embeddings & Vector Store 
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name = "rag_base",
    embedding_function=embedding,
)
vector_store.add_documents(documents=chunks)
print("\nVector store created successfully.")

# 4. Create Retriever
retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
print("\nRetriever created successfully.")

# 5.Graph State
class AgenticRAGState(MessagesState):
    query:str
    retrieved_docs:list[Document]
    context:str
    response:str
    
# 6.initialize llm
llm = ChatOpenAI(model="gpt-5-mini")

#======1st NODE ======
# retrieves relevant docs from the vector store
def retrieve(state:AgenticRAGState)-> dict:
    docs = retriever.invoke(state["query"])
    context = "\n\n".join([d.page_content for d in docs]) # joiing each paragraph onto one
    print(f"\nRetrieved {len(docs)} chunks.\n")
    return {
        "retrieved_docs" : docs,
        "context" : context
    }
    