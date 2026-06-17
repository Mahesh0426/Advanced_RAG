from langchain_classic.retrievers.document_compressors.chain_extract_prompt import prompt_template
from langgraph.graph import StateGraph,END,START,MessagesState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

from pathlib import Path
from typing import Literal
from pydantic import BaseModel

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
    collection_name="rag_conditional",
    embedding_function=embedding,
)
vector_store.add_documents(documents=chunks)
print("\nVector store created successfully.")

# 4. Create Retriever
retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
print("\nRetriever created successfully.")

# 5. llm intialization
llm =ChatOpenAI(model ="gpt-5-mini",temperature=0)

# 6. State Graph definition - shared memory of the graph.
class AgenticRAGState(MessagesState):
    query:str
    retrieved_docs:list[Document] | None
    context:str
    generation:str
    need_retrieval:bool

#======1st NODE======
# 7.a structure output schema for the routing decision
class RouteDecision(BaseModel):
    needs_retrieval:bool

# 7.b route_question node - it classifies whether the query needs documents for retrieval or not.
def route_question_node(state:AgenticRAGState) -> dict:
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system","Classify whether the following question requires retrieving information from a specialized document, or can be answered from your own general knowledge."),
            ("human","Question:{query}")
        ]
    )
    chain = prompt_template | llm.with_structured_output(RouteDecision) 
    decision = chain.invoke({"query":state["query"]})
    return {"need_retrieval": decision.needs_retrieval}

#======2nd NODE======
# 8. retrieved node - fetches relevant docs from the vector store.
def retrieve_node(state:AgenticRAGState) -> dict:
    docs = retriever.invoke(state["query"])
    
    if docs:
        context = "\n\n".join([doc.page_content for doc in docs])
    
    return {"retrieved_docs": docs, "context": context}
    
#======3rd NODE====== 
# 9. generate node - produce the final answer,with or without retrieve context
def generate_node(state:AgenticRAGState) -> dict:
    
    query = state["query"]
    context = state.get("context","")
    
    if context:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using only the context below. \n\nContext:\n{context}"),
            ("human","Question:{query}")
        ])
        response = (prompt_template | llm).invoke({"context":context,"query":query})
    else:
          prompt_template = ChatPromptTemplate.from_messages([
              ("system","Answer the question based on your general knowledge"),
              ("human","Question:{query}")
          ])
          response = (prompt_template | llm).invoke({"query":query})

    return {"generation" : response.content}
        


# ======= 10. The conditional ROUTER (routing function) =====   
# routing fn : it maps need_retrieval bool to the next node 
def route_after_classification(state: AgenticRAGState) -> Literal["retrieve_node","generate_node"]:
    return "retrieve_node" if state["need_retrieval"] else "generate_node" 
        
# 11. Graph defination
g = StateGraph(AgenticRAGState)
g.add_node("route_question_node",route_question_node)
g.add_node("retrieve_node",retrieve_node)
g.add_node("generate_node",generate_node)

# 12. edges
g.add_edge(START,"route_question_node")
g.add_conditional_edges("route_question_node",route_after_classification)
g.add_edge("retrieve_node", "generate_node")
g.add_edge("generate_node", END)

graph = g.compile()
save_path = base_dir / "agentic_rag_step02.png"
png_data = graph.get_graph().draw_mermaid_png()
with open(save_path, "wb") as f:
    f.write(png_data)
print("Graph saved successfully!")
    

# Query 1: domain-specific (should trigger retrieval)
domain_query = "What does the report say about EV adoption trajectories and oil demand displacement?"
result_domain = graph.invoke({"query": domain_query, "messages": []}) # messages - Holds the conversation history for multi-turn chat.

# Output: Query 1 (domain-specific)
print("=== QUERY 1 (domain-specific) ===")
print(f"Query          : {domain_query}")
print(f"needs_retrieval: {result_domain['need_retrieval']}")
retrieved = result_domain.get("retrieved_docs") or []
print(f"Retrieved docs : {len(retrieved)} docs")
print(f"\nGeneration:\n{result_domain['generation']}")


# Query 2: general knowledge (should skip retrieval)
general_query = "What is the capital of Australia?"
result_general = graph.invoke({"query": general_query, "messages": []})

# Output: Query 2 (general knowledge)
print("\n=== QUERY 2 (general knowledge) ===")
print(f"Query          : {general_query}")
print(f"needs_retrieval: {result_general['need_retrieval']}")
print(f"Retrieved docs : {result_general.get('retrieved_docs')}")
print(f"\nGeneration:\n{result_general['generation']}")
    
    
    

