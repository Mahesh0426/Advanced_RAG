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
    relevant_docs:List[Document]
    answer:str
    context: str


# Pydantic schema that defines the exact JSON structure the LLM must return.
# Inheriting from BaseModel enables automatic type validation and JSON schema
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
# 6.decide_retrieval - it decides whether to retrieve documents or not
def decide_retrieval_node(state: State) -> State:
    """1st node: decide retrieval need."""
    decision: RetrieveDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(question=state["question"])
    )
    
    print("\nRetrieval needed:", decision.should_retrieve)
    return {"need_retrieve":decision.should_retrieve}


# prompt template for direct generation node
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

# 
class RelevanceDecision (BaseModel):
    is_relevant:bool = Field(
        ...,
        description="True if the document helps answer the question ,else False"
    )
is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
     (
            "system",
            "You are judging document relevance.\n"
            "Return JSON that matches this schema:\n"
            "{{'is_relevant': boolean}}\n\n"
            "A document is relevant if it contains information useful for answering the question."
        ),
        (
            "human",
            "Question:\n{question}\n\nDocument:\n{document}"
        ),
    ]
)
relevance_llm = llm.with_structured_output(RelevanceDecision)

# ======= 4th Node======= 
# 10. is_relevant_node - it checks the relevance of the retrieved documents
def is_relevant_node(state: State):
    
    relevant_docs: List[Document] = []

    for doc in state["docs"]:
        decision: RelevanceDecision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"],
                document=doc.page_content
            )
        )

        if decision.is_relevant:
            relevant_docs.append(doc)

    return {"relevant_docs": relevant_docs}


rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a business RAG assistant.\n"
            "Answer the user's question using ONLY the provided context.\n"
            "If the context does not contain enough information, say:\n"
            "'No relevant document found.'\n"
            "Do not use outside knowledge.\n"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Context:\n{context}\n"
        ),
    ]
)

 # ======= 5th Node =======
 # generate_from_context - it generates the answer from the relevant documents
def generate_from_context_node(state: State):
    # Stuff relevant docs into one block
    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", [])]
    ).strip()

    if not context:
        return {"answer": "No relevant document found.", "context": ""}

    out = llm.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"],
            context=context
        )
    )
    return {"answer": out.content, "context": context}

 # ======= 6th Node =======
 # 12. no_relevant_docs - it handles the case where no relevant documents are found
def no_relevant_docs_node(state: State):
    return {"answer": "No relevant document found.", "context": ""}

# ======= The conditional router (routing finction) =====
# 13.route_after_decide - It decides which node to call next based on the retrieval decision.
def route_after_decide(state: State) -> Literal["generate_direct","retrieve"]:
    """4th node: Conditional router. Decide next node based on retrieval decision."""
    if state["need_retrieve"]:
        print("Route to: retrieve")
        return "retrieve"
    else:
        print("Route to: generate_direct")
        return "generate_direct"

# 14.route_after_relevance - It decides which node to call next based on the relevance decision.
def route_after_relevance(state: State) -> Literal["generate_from_context", "no_relevant_docs"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_relevant_docs"

# 14. build the RAG graph 
g = StateGraph(State)

# 15. add nodes to the graph
g.add_node("decide_retrieval",decide_retrieval_node)
g.add_node("generate_direct",generate_direct_node)
g.add_node("retrieve",retrieve_node)
g.add_node("is_relevant",is_relevant_node)
g.add_node("generate_from_context",generate_from_context_node)
g.add_node("no_relevant_docs",no_relevant_docs_node)


#16. edges
g.add_edge(START,"decide_retrieval")
g.add_conditional_edges(
    "decide_retrieval",route_after_decide,
    {
        "generate_direct":"generate_direct",
        "retrieve":"retrieve"
    }
)
g.add_edge("generate_direct",END)
g.add_edge("retrieve","is_relevant")
g.add_conditional_edges(
    "is_relevant",route_after_relevance,
    {
        "generate_from_context":"generate_from_context",
        "no_relevant_docs":"no_relevant_docs"
    }
)
g.add_edge("generate_from_context",END)
g.add_edge("no_relevant_docs",END)


workflow = g.compile()   
png_data = workflow.get_graph().draw_mermaid_png()
with open("self_rag_step02.png", "wb") as f:
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



print("\n" + "relevant docs: ")
for doc in result['relevant_docs']:
    print(doc.page_content)
    print("*"*100)

# ================== 2nd test case =================
print("\n" + "="*80)
print("TEST CASE 2: With general knowledge question\n")
result = workflow.invoke({"question":"What is the refund policy of NexaAI"})

print("\n" + "checking need retrieval:")
print(result["need_retrieve"])

print("\n--- Final Answer ---")
print(result["answer"])



# print("\n" + "checking docs")
# print(result["docs"])

# START
# │
# ▼
# decide_retrieval
# │
# ▼
# route_after_decide
# │
# ├───────────────────────────────┐
# │                               │
# ▼                               ▼
# generate_direct               retrieve
# │                               │
# ▼                               ▼
# END                        is_relevant
#                                 │
#                                 ▼
#                                 END