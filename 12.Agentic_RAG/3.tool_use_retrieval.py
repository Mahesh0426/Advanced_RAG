import os
import operator
from pathlib import Path
from typing import Annotated, Literal
from pydantic import BaseModel
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()

base_dir = Path(__file__).resolve().parent
documents_dir = base_dir / "documents"
pdf_path = documents_dir / "evs_oil_price_shock.pdf"

# 1. Load the PDF
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()
print(f"Loaded {len(docs)} pages from the PDF.")

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"\nSplit into {len(chunks)} chunks.")

# 3. Embeddings & Vector Store
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="rag_tool_use",
    embedding_function=embedding,
)
vector_store.add_documents(documents=chunks)
print("\nVector store created successfully.")

# 4. initializing model
llm = ChatOpenAI(model="gpt-5-mini",temperature=0)
agent_llm = ChatOpenAI(model="gpt-4o-mini")

# 6. State Graph definition - shared memory of the graph.
class AgenticRAGState(MessagesState):
    query:str
    retrieved_docs:Annotated[list[Document], operator.add]
    context:Annotated[str,operator.add]
    generation:str
    needs_retrieval:bool
    

# 7.a structure output schema for the routing decision
class RouteDecision(BaseModel):
    needs_retrieval:bool
 
# ----------------1st TOOL -----------------
# @tool decorator makes the function tool-capable for the graph.
@tool(response_format="content_and_artifact")
def vector_store_search(query:str,k:int=3):
    """Search the vector store for relevant document passages.
    Adjust k (default 3) to retrieve more or fewer passages."""
    
    retriever = vector_store.as_retriever(search_type = "similarity",search_kwargs={"k":k})
    docs = retriever.imvoke(query)
    
    context = "\n\n## Vector Store Results\n\n" + "\n\n".join([d.page_content for d in docs]) # merge into one context string for llm
    return context,docs


# ----------------2nd TOOL -----------------
@tool(response_format="content_and_artifact")
def web_search(query:str, max_results:int=3):
    """Search the web for current or real-time information.
    Adjust max_results (default 3) to control how many results are returned."""
    
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query=query,max_results=max_results)
    
    docs = [
        Document(
            page_content= r["content"],
            metadata = {"source": r["url"],"title":r.get("title","")}
        )
        for r in response.get("results",[])
    ]
    content = "\n\n## web Search Results\n\n" + "\n\n".join([d.page_content for d in docs])
    return content,docs
    

    
    
    
    

    


    


