from typing import List, TypedDict
from pydantic import BaseModel
import re
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from  langgraph.graph import StateGraph,END,START
from dotenv import load_dotenv

load_dotenv()

base_dir = os.path.dirname(__file__)
pdf_path1 = os.path.join(base_dir, "c_rag.pdf")
pdf_path2 = os.path.join(base_dir, "c_rag2.pdf")

# 1. Load the PDF
loader = PyPDFLoader(pdf_path1).load()
loader2 = PyPDFLoader(pdf_path2) .load()
pages = loader + loader2

print(f"Loaded {len(pages)} page(s) from the PDF.")

# 2. Split Documents into Chunks
chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(pages)
for d in chunks:
    d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
print(f"\nSplit into {len(chunks)} chunk(s).\n")
    

# 3. Embeddings & Vector Store
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.from_documents(chunks, embedding)
print("\nVector store created successfully.")

# 4. Create Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

UPPER_TH = 0.7  # a chunk scoring above this is considered highly relevant (CORRECT)
LOWER_TH = 0.3  # a chunk scoring below this is considered noise (INCORRECT)
                # anything between the two thresholds is AMBIGUOUS

# 5. State defination - This is the shared memory that flows through the entire StateGraph. very node reads from and writes to this dict.
class State(TypedDict):
    question: str
    docs: List[Document]
    
    good_docs: List[Document]  # chunks that scored above LOWER_TH — survive into refine
    verdict: str               # overall judgment: "CORRECT", "INCORRECT", or "AMBIGUOUS"
    reason: str                # human-readable explanation of why that verdict was reached
    

    strips: List[str]  #the raw text chunks extracted from retrieved docs, broken down into smaller pieces (strips)
    kept_strips: List[str] # after a relevance grading step, only the strips that are relevant to the question survive here
    refined_context: List[str] # after a rewriting or refinement step, improved strips are stored here
    web_search:List[Document] # web results fetched by Tavily when verdict is INCORRECT or AMBIGUOUS
    answer:str
    web_query:str

# 6. Retrieve node - 1st NODE. It takes the question from the state and returns the retrieved chunks.
def retrieve(state):
    print("\n--- retrieve node ---")
    q=state["question"]
    print("Question:", q)
    docs = retriever.invoke(q)
    print(f"Retrieved {len(docs)} chunks\n")
    return {"docs": docs}

#7. Doc-level scoring node — returns a numeric score (0–1) for each retrieved chunk.
class DocEvalScore(BaseModel):
    score:float 
    reason:str

#system prompt
doc_eval_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict relevance evaluator for RAG.\n"
     "You will be given ONE retrieved chunk and a question.\n"
     "Return a relavance score [0.0, 1.0]\n"
     "- 1.0: chunk alone is sufficient to answer the fully/mostly\n"
     "- 0.0: chunk is irrelevant\n"
     "Be conservative with high scores.\n"
     "Also return a short reason.\n"
     "Output JSON only"
     ),
    ("human","Question: {question}\n\nChunk:\n {chunk}")
])
# chains prompt → LLM → parsed DocEvalScore object
doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)

#8. Doc-level filtering node - 1st node 
def eval_each_doc_node(state:State)->State:
    print("\n--- eval each doc node ---")
    q=state["question"]
    print("Question:", q)
    
    scores:List[float] = []
    reasons:List[str] = []
    good_doc:List[Document] = []
    
    # Score every retrieved chunk individually — one LLM call per doc (4 calls total)
    for doc in state["docs"]:
        out = doc_eval_chain.invoke({"question":q,"chunk":doc.page_content})
        scores.append(out.score)
        reasons.append(out.reason)
        
        # ----1.for CORRECT case we will refine only docs with score > LOWER_TH-----
        if out.score > LOWER_TH:
            good_doc.append(doc)
    
    # ----2.CORRECT if al least one doc > UPPER_TH----
    if any(s > UPPER_TH for s in scores): # "Go through every score in the list one by one — if even a single score is above 0.7, stop immediately and return True.
        return{
            "good_docs":good_doc,
            "verdict":"CORRECT",
            "reason":f"At least one retrieved chunk scored > {UPPER_TH}",
        }
    
    # -----3.INCORRECT if all docs < LOWER_TH
    if len(scores)> 0 and all(s < LOWER_TH for s in scores): #If the scores list is not empty AND every single score is below 0.3, then all retrieved chunks are noise
        why = "No chunks was sufficient"
        return{
            "good_docs":[],
            "verdict":"INCORRECT",
            "reason":f"All retrieved chunks scored <  {LOWER_TH}. {why}",
        }

    
    # -----4.AMBIGUOUS  any thing in between
    why = "Mixed relevance signals"
    return{
        "good_docs":good_doc,
        "verdict":"AMBIGUOUS",
        "reason":f"No chunks scored > {UPPER_TH}, but not all were < {LOWER_TH}. {why}",
    }
    
    
# 9. sentence-level decomposer - return list of raw text strips
# Breaks a raw chunk of text into individual sentences for fine-grained relevance grading.
def decompose_to_sentences(text:str) -> List[str]:
    text = re.sub(r'\s+', " ", text).strip() # normalize whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]  # Drop any sentence shorter than 20 chars

#10. Pydantic schema for the LLM's structured output.
# Forces the LLM to return a single boolean — no free text, no ambiguity.
class KeepOrDrop(BaseModel):
    keep:bool

# Prompt that instructs the LLM to act as a binary relevance judge.
firter_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict Relevance filter."
     "Return keep=true only if the sentence directly helps answer the question.\n"
     "Use ONLY the  sentence. Output JSON only."
     ),
    ("human","Question: {question}\n\nSentence:\n {sentence}")
])

# Chains the prompt → LLM → structured output parser in one callable.
filter_chain = firter_prompt | llm.with_structured_output(KeepOrDrop) 

#11. refining 2nd NODE. (decompose - filter - recompose)
def refine(state: State)-> State:
    print("\n--- refine node ---")
    q = state["question"]
    print("Question:", q)
    
    if state.get("verdict") == "CORRECT":
        #combine retrieved docs into one context string
        context = "\n\n".join(d.page_content for d in state["good_docs"]).strip()
    else:
       context = " \n\n".join(d.page_content for d in state["web_search"]).strip()
    
    # ── Stage 1: DECOMPOSE ────────────────────────────────────────────────────
    # Split the merged context into individual sentences.
    strips = decompose_to_sentences(context)
    print(f"\nDecomposed into {len(strips)} sentence strips")

    # ── Stage 2: FILTER ───────────────────────────────────────────────────────
    #  Pass each sentence through the LLM judge (filter_chain).Only sentences where keep=True survive into the next stage.
    kept:List[str] = []
    for strip in strips:
        if filter_chain.invoke({"question":q,"sentence":strip}).keep:
            kept.append(strip)
    print(f"\nKept {len(kept)} out of {len(strips)} strips")
    
  # ── Stage 3: RECOMPOSE ────────────────────────────────────────────────────
   # Glue the surviving sentences back into a clean context string.
    refined_context = "\n\n".join(kept).strip()
    
    return{
        "strips":strips,  # all sentences before filtering
        "kept_strips":kept,  # sentences that passed the relevance check
        "refined_context":refined_context # final clean context for generation
    }

class WebQuery(BaseModel):
    query:str

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "rewrite the user question into a web search query composed of keywords.\n"
     "Rules:\n"
     "- keep it short (6-14 words).\n"
     "- If the question implies a recency, (for example, recents/latest/last month), add a constant like,in last 30 days.\n"
     "- Do NOT answer the question.\n"
     "- Return JSON with a single key: query."
     ),
    ("human", "Question: {question}")
])
rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)


#Rewrite query node
def rewrite_query_node(state:State)->State:
    print("\n--- rewrite query node ---")
    out = rewrite_chain.invoke({"question": state["question"]})
    print("Rewritten Query:", out.query)
    return {"web_query": out.query}


# Web search node  - 3rd node
tavily = TavilySearchAPIRetriever(max_results=5)

def web_search_node(state:State)->State:
    print("\n--- web search node ---")
    q=state.get("web_query") or state["question"]
    print("Web Query:", q)
    results = tavily.invoke(q)
    print("\n Web Search Results\n")
    print(results)
    
    web_docs = []
    for r in results or []:
        url = r.metadata.get("url", "")
        title = r.metadata.get("title", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT: {r.page_content}\n\n"
        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))
    return{"web_search":web_docs}
    
        
#Prompt for the final generation step.
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful ML tutor. Answer ONLY using the provided refined bullets.\n" 
     "If the bullets are empty or insufficient, say: 'I don't know based on the provided books'"
     ),
    ("human", "Question: {question}\n\nRefined context:\n{refined_context}")
])

#12. generate node - 4th NODE
def generate(state:State)->State:
   print("\n--- generate node ---")
   out = (answer_prompt | llm).invoke({"question":state["question"],"refined_context":state["refined_context"]})
   return{"answer" : out.content}

#ambiguous_node 5th node
def ambiguous_node(state:State)->State:
    return{ "answer":f"AMBIGUOUS: {state['reason']}"}

#the routing function -  decides where the flow should go next
def route_after_eval(state:State)->str:
    if state["verdict"] == "CORRECT":
        return "refine"
    elif state["verdict"] == "INCORRECT":
        return "rewrite_query"
    else:
        return "ambiguous"

    
# 11. build the RAG graph - START → retrieve →refine → generate → END
g = StateGraph(State)
g.add_node("retrieve",retrieve)
g.add_node("eval_each_doc",eval_each_doc_node)
g.add_node("rewrite_query",rewrite_query_node)
g.add_node("web_search",web_search_node)
g.add_node("refine",refine)
g.add_node("generate",generate)
g.add_node("ambiguous",ambiguous_node)


g.add_edge(START,"retrieve")
g.add_edge("retrieve","eval_each_doc")
g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine": "refine",
        "rewrite_query": "rewrite_query",
        "web_search": "web_search",
        "ambiguous": "ambiguous"
    }
)
#INCORRECT path : re-write -> web_search -> refine -> generate
g.add_edge("rewrite_query","web_search")
g.add_edge("web_search","refine")
g.add_edge("refine","generate")
g.add_edge("generate",END)
g.add_edge("ambiguous",END)


workflow = g.compile()

res = workflow.invoke({
    "question":"Latest news about quantum computing breakthroughs",
    "docs":[],
    "good_docs":[],
    "verdict":[],
    "reason":[],
    "strips": [],
    "kept_strips" :[],
    "web_search":[],
    "refined_context":"",
    "answer": ""
    })


print("\nVERDICT:", res["verdict"])
print("\nREASON:", res["reason"])
print("\n-------Final answer: ------")
print(res["answer"])

print("\n re-written query:")
print(res.get("web_query", "N/A (verdict was not INCORRECT, rewrite not triggered)"))

# print("\n--- All strips ---")
# for i, strip in enumerate(res["strips"], 1):
#     print(f"  [{i}] {strip}")

# print(f"\n--- Kept strips ({len(res['kept_strips'])})/{len(res['strips'])}) ---")
# for i, strip in enumerate(res["kept_strips"], 1):
#     print(f"  [{i}] {strip}")

# AI news for last week
# What are attention mechanisms and why they are importnt  in current models?
