from langgraph.graph import StateGraph,END,START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

from typing import List,TypedDict,Literal
from pathlib import Path
from pydantic import BaseModel, Field

load_dotenv()

base_dir = Path(__file__).resolve().parent
documents_dir = base_dir / "documents"

pdf_path1 = documents_dir / "Company_Policies.pdf"
pdf_path2 = documents_dir / "Company_Profile.pdf"
pdf_path3 = documents_dir / "Product_and_Pricing.pdf"

# 1. Load the PDF
docs: List = (
    PyPDFLoader(str(pdf_path1)).load()
    + PyPDFLoader(str(pdf_path2)).load()
    + PyPDFLoader(str(pdf_path3)).load()
)
print(f"Loaded {len(docs)} page(s) from the PDF.")

# 2. Split Documents into chunks
chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150).split_documents(docs)
print(f"\nSplit into {len(chunks)} chunks.")

# 3. Embeddings & Vector Store
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.from_documents(chunks, embedding)
print("\nVector store created successfully.")

# 4. Create Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print("\nRetriever created successfully.")

# 5. Define the LLM
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
print("\nLLM created successfully.")

#5. Graph state - it is the shared memory that flows through the entire StateGraph. Every node reads from it and writes to it
class State(TypedDict):
    question:str
    need_retrieve:bool
    docs:List[Document]
    answer:str

# 6. Decide retrieval node - it decides whether to retrieve documents or not.
class RetrieveDecision(BaseModel):
    should_retrieve:bool = Field(
        ...,
        description="True if external documents are needed to answer reliably ,else False"
        )

# prompt template for decide retrieval node
decide_retrieval_prompt = ChatPromptTemplate(
        [
            (
            "system",
            "You decide whether retrieval is needed.\n"
            "Return JSON that matches this schema:\n"
            "{{'should_retrieve': boolean}}\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts, citations, or info likely not in the model.\n"
            "- should_retrieve=False for general explanations, definitions, or reasoning that doesn't need sources.\n"
            "- If unsure, choose True."
        ),
        ("human", "Question: {question}"),
        ]
    )

# create llm with structured output - this is used to get the JSON output from the LLM
should_retrieve_llm = llm.with_structured_output(RetrieveDecision)

# ======= 1st Node=======
# 7. decide_retrieval - it decides whether to retrieve documents or not
def decide_retrieval_node(state: State) -> State:
    """1st node: decide retrieval need."""
    decision: RetrieveDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(question=state["question"])
    )
    
    print("\nRetrieval needed:", decision.should_retrieve)
    return {"need_retrieve":decision.should_retrieve}

direct_generation_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "Answer the question using only your general knowledge.\n"
            "Do NOT assume access to external documents.\n"
            "If you are unsure or the answer requires specific sources, say:\n"
            "'I don't know based on my general knowledge.'"
        ),
        ("human", "{question}"),
    ]
)

# ======= 2nd Node======= 
# 8. geberate_direct - Answer directly without retrieval
def generate_direct_node(state: State) -> State:
    """2nd node: Answer directly without retrieval."""
    out = llm.invoke(
        direct_generation_prompt.format_messages(
            question=state["question"]
        )
    )
    return {
        "answer": out.content
    }

# ======= 3rd Node======= 
# 9. retrieve node - it retrieves documents from the vector store
def retrieve_node(state: State) -> State:
    """3rd node: Retrieve documents from vector store."""
    docs: List[Document] = retriever.invoke(state["question"])
    print(f"Retrieved {len(docs)} chunks.\n")
    return {"docs": docs}

# ======= The conditional router (routing finction) =====
# 10.route_after_decide - It decides which node to call next based on the retrieval decision.
def route_after_decide(state: State) -> Literal["generate_direct","retrieve"]:
    """4th node: Conditional router. Decide next node based on retrieval decision."""
    if state["need_retrieve"]:
        print("Route to: retrieve")
        return "retrieve"
    else:
        print("Route to: generate_direct")
        return "generate_direct"

# 11. build the RAG graph  START → decide_retrieval → [route] → generate_direct → END
#                                                             ↘ retrieve        → END
g = StateGraph(State)

# 12. add nodes to the graph
g.add_node("decide_retrieval",decide_retrieval_node)
g.add_node("generate_direct",generate_direct_node)
g.add_node("retrieve",retrieve_node)

#13. edges
g.add_edge(START,"decide_retrieval")
g.add_conditional_edges(
    "decide_retrieval",route_after_decide,
    {
        "generate_direct":"generate_direct",
        "retrieve":"retrieve"
    }
)
g.add_edge("generate_direct",END)
g.add_edge("retrieve",END) # temporary  END for retrieval path


workflow = g.compile() 
  
png_data = workflow.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)
print("Graph saved successfully!")

result = workflow.invoke({
    "question":"who is CEO of Nexa AI ?",
    "need_retrieve":False,
    "docs": [],
    "answer": ""
    })

print("\n" + "="*80)
print("TEST CASE 1: With document-dependent question\n")
print("\n" + "checking need retrieval:")
print(result["need_retrieve"])
print(result["answer"])



print("\n" + "checking docs")
# print(result["docs"])

# ================== 2nd test case =================
print("\n" + "="*80)
print("TEST CASE 2: With general knowledge question\n")
result = workflow.invoke({"question":"what is machine learning"})

print("\n" + "checking need retrieval:")
print(result["need_retrieve"])

print("\n--- Final Answer ---")
print(result["answer"])



# print("\n" + "checking docs")
# print(result["docs"])


# # START
# #   └─► decide_retrieval
# #           ├─ need_retrieve=False ──► generate_direct ──► END
# #           └─ need_retrieve=True  ──► retrieve        ──► END (temporary)