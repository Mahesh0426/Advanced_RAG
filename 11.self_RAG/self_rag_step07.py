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
docs:List = (
    PyPDFLoader(str(pdf_path1)).load()
    + PyPDFLoader(str(pdf_path2)).load()
    + PyPDFLoader(str(pdf_path3)).load()
)
print(f"Loaded {len(docs)} pages(s) from the PDF.")

# 2. Split documents into chunks
chunks  = RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=150).split_documents(docs)
print(f"\nSplit into {len(chunks)} chunks.")

# 3. Embeddings & Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.from_documents(chunks, embeddings)
print("\nVector store created successfully...")

# 4. Create Retriever
retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
print("\nRetriever created successfully...")

# 5. Define the LLM
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
print("\nLLM created successfully...")

#5. Graph state - it is the shared memory that flows through the entire StateGraph. Every node reads from it and writes to it
class State(TypedDict):
        question:str
        need_retrieve:bool
        docs:List[Document]
        relevant_docs:List[Document]
        answer:str
        context: str
        #post-geberation verification
        issup:Literal["fully_supported", "partially_supported","not_supported"]
        evidence:List[str]
        retries:int
        isuse: Literal["useful", "not_useful"]
        use_reason: str
        
        # ✅ NEW: what we actually send to vector retriever
        retrieval_query: str
        rewrite_tries: int


# ======= 1st Node=======  
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

# 6.decide_retrieval_node - it decides whether to retrieve documents or not
def decide_retrieval_node(state: State) -> State:
    """1st node: decide retrieval need."""
    decision:RetrieveDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(question=state["question"])
    )
    print("\nIs Retrieval needed:", decision.should_retrieve)
    return {"need_retrieve": decision.should_retrieve}


# ======= 2nd Node=======
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

# 7. generate_direct_node - Answer directly without retrieval
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
    q=state.get("retrieval_query") or state["question"]
    docs=retriever.invoke(q)
    print(f"\nRetrieved {len(docs)} chunks.\n")
    return {"docs": docs}


# ======= 4th Node======= 
# Pydantic schema that defines the exact JSON structure the LLM must return.
class RelevanceDecision (BaseModel):
    is_relevant:bool = Field(
        ...,
        description="True if the document helps answer the question ,else False"
    )

# prompt template for relevance check node
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


# ======= 5th Node======= 
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

#11. generate_from_context_node - it generates the answer from the relevant documents
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
 # 12. no_answer_found - it handles the case where no answer is found
def no_answer_found_node(state: State):
    return {"answer": "No answer found.", "context": ""}


# ======= 7th Node====== 
# IsSup verify + revise loop
# Pydantic schema that defines the exact JSON structure the LLM must return.
class IsSUPDecision(BaseModel):
    issup:Literal['fully_supported','partially_supported','not_supported']
    evidence:List[str] = Field(default_factory=list)
    
# is_sup_prompt
issup_prompt = ChatPromptTemplate.from_messages(
     [
         (
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
            "Return JSON with keys: issup, evidence.\n"
            "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "How to decide issup:\n"
            "- fully_supported:\n"
            "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce\n"
            "  any qualitative/interpretive words that are not present in CONTEXT.\n"
            "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to,\n"
            "  supports professional development, best-in-class, employee-first, etc.)\n\n"
            "- partially_supported:\n"
            "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative\n"
            "  phrasing not explicitly stated in CONTEXT (e.g., calling policies 'culture', saying leave is 'generous',\n"
            "  or inferring outcomes like 'supports professional development').\n\n"
            "- no_support:\n"
            "  The key claims are not supported by CONTEXT.\n\n"
            "Rules:\n"
            "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported.\n"
            "- If the answer is mostly unrelated to the question or unsupported, choose no_support.\n"
            "- Evidence: include up to 3 short direct quotes from CONTEXT that support the supported parts.\n"
            "- Do not use outside knowledge."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Context:\n{context}\n"
        ),
         
     ]
 )
issup_llm = llm.with_structured_output(IsSUPDecision)

# 13. is_supported_node - it checks the support level of the answer | it find  hallucination
def is_sup_node(state:State):
    decision:IsSUPDecision = issup_llm.invoke(
        issup_prompt.format_messages(
            question = state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {"issup":decision.issup, "evidence":decision.evidence}


# ======== 8th Node====== 
# 14. accept_answer_node - It accepts the answer as it is.
def accept_answer_node(state: State):
    return {}  # keep answer as-is

# ======== 9th Node====== 
revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "You must output based on the following format:\n\n"
            "FORMAT (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
            "- Do NOT explain anything.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current Answer:\n{answer}\n\n"
            "CONTEXT:\n{context}"
        ),
    ]
)

#15.revise_answer_node - it revises the answer based on the support level of the answer.
def revise_answer_node(state: State):
    out = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    
    return {
        "answer": out.content,
        "retries": state.get("retries", 0) + 1,  # ✅ increment
    }
   
   
# ======== 10th Node======  
#  Pydantic schema that defines the exact JSON structure the LLM must return.
class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")

# prompt template for usefulness check node
isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
            "Goal:\n"
            "- Decide if the answer actually addresses what the user asked.\n\n"
            "Return JSON with keys: isuse, reason.\n"
            "isuse must be one of: useful, not_useful.\n\n"
            "Rules:\n"
            "- useful: The answer directly answers the question or provides the requested specific info.\n"
            "- not_useful: The answer is generic, off-topic, or only gives related background without answering.\n"
            "- Do NOT use outside knowledge.\n"
            "- Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'\n"
            "- Keep reason to 1 short line."
        ),
        (
            "human",
            "Question:\n{question}\n\nAnswer:\n{answer}"
        ),
    ]
)
isuse_llm = llm.with_structured_output(IsUSEDecision)

#17. is_use_node - it checks the usefullness of the answer 
def is_use_node(state: State):
    decision: IsUSEDecision = isuse_llm.invoke(
        isuse_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
        )
    )
    return {"isuse": decision.isuse, "use_reason": decision.reason}


# ======== 11th Node====== 
class RewriteDecision(BaseModel):
    retrieval_query:str = Field(
        ...,
        description="Rewritten query optimized for vector retrieval, against internal company PDFs."
    )
    
rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
            "Rules:\n"
            "- Keep it short (6–16 words).\n"
            "- Preserve key entities (e.g., NexaAI, plan names).\n"
            "- Add 2–5 high-signal keywords that likely appear in policy/pricing docs.\n"
            "- Remove filler words.\n"
            "- Do NOT answer the question.\n"
            "- Output JSON with key: retrieval_query\n\n"
            "Examples:\n"
            "Q: 'Do NexaAI plans include a free trial?'\n"
            "-> {{'retrieval_query': 'NexaAI free trial duration trial period plans'}}\n\n"
            "Q: 'What is NexaAI refund policy?'\n"
            "-> {{'retrieval_query': 'NexaAI refund policy cancellation refund timeline charges'}}"
        ),
        (
            "human",
            "QUESTION:\n{question}\n\n"
            "Previous retrieval query:\n{retrieval_query}\n\n"
            "Answer (if any):\n{answer}"
        ),
    ]
)
rewrite_llm = llm.with_structured_output(RewriteDecision)

# rewrite_question_node - it rewrite the question optimized for vector retrieval
def rewrite_question_node(state:State):
    decision:RewriteDecision = rewrite_llm.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", "")
        )
        
    )
    return{
        "retrieval_query":decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        # optional 
        "docs":[],
        "relevant_docs":[],
        "context":"",
    }
    


# ======= 16. The conditional ROUTER (routing function) =====
MAX_RETRIES = 10
# 16.a route_after_issup - It decides which node to call next based on the support level of the answer.
def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
    if state.get("issup") == "fully_supported":
        return "accept_answer"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"
    return "revise_answer"

# 16.b route_after_decide - It decides which node to call next based on the retrieval decision.
def route_after_decide(state: State) -> Literal["generate_direct","retrieve"]:
    """4th node: Conditional router. Decide next node based on retrieval decision."""
    if state["need_retrieve"]:
        print("Route to: retrieve")
        return "retrieve"
    else:
        print("Route to: generate_direct")
        return "generate_direct"

# 16.c route_after_relevance - It decides which node to call next based on the relevance decision.
def route_after_relevance(state: State) -> Literal["generate_from_context", "no_answer_found"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"

# 16.d route_after_isuse - It decides whether to accept or reject the answer based on the usefullness decision.
def route_after_isuse(state: State) -> Literal["END", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"
    return "no_answer_found"

# ============================================================

# 17. build the RAG graph 
g = StateGraph(State)

# 18. add nodes to the graph
g.add_node("decide_retrieval",decide_retrieval_node)
g.add_node("generate_direct",generate_direct_node)
g.add_node("retrieve",retrieve_node)
g.add_node("is_relevant",is_relevant_node)
g.add_node("generate_from_context",generate_from_context_node)
g.add_node("no_answer_found",no_answer_found_node)
g.add_node("is_sup",is_sup_node)
g.add_node("accept_answer", accept_answer_node) 
g.add_node("revise_answer", revise_answer_node) 
g.add_node("is_use", is_use_node) 
g.add_node("rewrite_question",rewrite_question_node) # ✅ NEW: re-write question for better retrieval

#19. edges
g.add_edge(START, "decide_retrieval")

g.add_conditional_edges(
    "decide_retrieval", route_after_decide,
    {
        "generate_direct": "generate_direct", 
        "retrieve": "retrieve"
    },
)
g.add_edge("generate_direct", END)

# retrieve --> relevance --> (generate_from_context | no_answer_found)
g.add_edge("retrieve", "is_relevant")
g.add_conditional_edges(
    "is_relevant", route_after_relevance,
    {
        "generate_from_context": "generate_from_context", 
        "no_answer_found": "no_answer_found"
    },
)
# If no answer found, end
g.add_edge("no_answer_found", END)

 # Generate -> IsSUP -> (IsUSE | revise) loop
g.add_edge("generate_from_context", "is_sup")
g.add_conditional_edges(
    "is_sup", route_after_issup,  # fully_supported -> accept_answer else revise_answer
    {
        "accept_answer": "is_use", 
        "revise_answer": "revise_answer"
    },
)
# revise then re-check support
g.add_edge("revise_answer", "is_sup")  # 🔁 loop back to verify
#  IsUSE routing
# useful --> END ,  not_useful --> rewrite_question --> retrieve ... loop
# give_up --> no_answer_found --> END
g.add_conditional_edges(
    "is_use", route_after_isuse, # useful -> END else no_answer_found
    {
        "END": END,
        "rewrite_question": "rewrite_question",
        "no_answer_found": "no_answer_found",
    },
)
# rewrite -> retrieve -> relevance -> ...
g.add_edge("rewrite_question", "retrieve")




workflow = g.compile()   
png_data = workflow.get_graph().draw_mermaid_png()
with open("self_rag_step07.png", "wb") as f:
    f.write(png_data)
print("Graph saved successfully!")

result = workflow.invoke(
    {
        # "question": "What is refund policy of NexaAI.",
        "question": "Describe NexaAI’s company culture.",
        "retrieval_query": "What is the refund policy of NexaAI.",
        "rewrite_tries":0,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "",
        "evidence": [],
        "retries": 0,
        "isuse":"not_useful",
        "use_reason":"",
    },
    config={"recursion_limit": 80},  # allow revise → verify loops
)



# -----------------------------
# Debug / inspection output (clean + complete)
# -----------------------------
print("\n===== RAG EXECUTION RESULT =====\n")

print("Question:", result.get("question"))
print("Need Retrieval:", result.get("need_retrieval"))

# If you added these counters/fields in your State:
print("Rewrite tries (retrieval):", result.get("rewrite_tries", 0))
print("Support revise tries:", result.get("retries", 0))

print("\nRetrieval:")
print("  Total retrieved docs:", len(result.get("docs", []) or []))
print("  Relevant docs:", len(result.get("relevant_docs", []) or []))

# Optional: show sources/pages for relevant docs
relevant_docs = result.get("relevant_docs", []) or []
if relevant_docs:
    print("\nRelevant docs (source/page):")
    for i, d in enumerate(relevant_docs, 1):
        src = (d.metadata or {}).get("source", "unknown")
        page = (d.metadata or {}).get("page", None)
        title = (d.metadata or {}).get("title", "")
        extra = f", title={title}" if title else ""
        if page is not None:
            print(f"  {i}. source={src}, page={page}{extra}")
        else:
            print(f"  {i}. source={src}{extra}")

print("\nVerification (IsSUP):")
print("  issup:", result.get("issup"))
evidence = result.get("evidence", []) or []
if evidence:
    print("  evidence:")
    for e in evidence:
        print("   -", e)
else:
    print("  evidence: (none)")

print("\nUsefulness (IsUSE):")
print("  isuse:", result.get("isuse"))
print("  reason:", result.get("use_reason", ""))

print("\nFinal Answer:")
print(result.get("answer"))

print("\n===============================\n")
