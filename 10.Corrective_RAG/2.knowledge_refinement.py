from typing import List, TypedDict
from pydantic import BaseModel
import re
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
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

# 5. State defination - This is the shared memory that flows through the entire StateGraph. very node reads from and writes to this dict.
class State(TypedDict):
    question: str
    docs: List[Document]

    strips: List[str]  #the raw text chunks extracted from retrieved docs, broken down into smaller pieces (strips)
    kept_strips: List[str] # after a relevance grading step, only the strips that are relevant to the question survive here
    refined_context: List[str] # after a rewriting or refinement step, improved strips are stored here

    answer:str

# 6. Retrieve node - 1st NODE. It takes the question from the state and returns the retrieved chunks.
def retrieve(state):
    print("\n--- retrieve node ---")
    q=state["question"]
    print("Question:", q)
    docs = retriever.invoke(q)
    print(f"Retrieved {len(docs)} chunks\n")
    return {"docs": docs}

# 7. sentence-level decomposer - return list of raw text strips
# Breaks a raw chunk of text into individual sentences for fine-grained relevance grading.
def decompose_to_sentences(text:str) -> List[str]:
    text = re.sub(r'\s+', " ", text).strip() # normalize whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]  # Drop any sentence shorter than 20 chars

#8. Pydantic schema for the LLM's structured output.
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

#8. refining 2nd NODE. (decompose - filter - recompose)
def refine(state: State)-> State:
    print("\n--- refine node ---")
    q = state["question"]
    print("Question:", q)
    
    ## Merge all 4 retrieved docs into one single text block for decomposition
    context = "\n\n".join(doc.page_content for doc in state["docs"])
    
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
    
#10. Prompt for the final generation step.
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful ML tutor. Answer ONLY using the provided refined bullets.\n" 
     "If the bullets are empty or insufficient, say: 'I don't know based on the provided books'"
     ),
    ("human", "Question: {question}\n\nRefined context:\n{refined_context}")
])

#10. generate node - 3rd NODE
def generate(state:State)->State:
   print("\n--- generate node ---")
   out = (answer_prompt | llm).invoke({"question":state["question"],"refined_context":state["refined_context"]})
   return{"answer" : out.content}
    
    
# 11. build the RAG graph - START → retrieve →refine → generate → END
g = StateGraph(State)
g.add_node("retrieve",retrieve)
g.add_node("refine",refine)
g.add_node("generate",generate)

g.add_edge(START,"retrieve")
g.add_edge("retrieve","refine")
g.add_edge("refine","generate")
g.add_edge("generate",END)

workflow = g.compile()

res = workflow.invoke({
    "question":"what is corrective rag?",
    "docs":[],
    "strips": [],
    "kept_strips" :[],
    "refined_context":"",
    "answer": ""
    })
print("\n-------Final answer: ------")
print(res["answer"])

    

print("\n--- All strips ---")
for i, strip in enumerate(res["strips"], 1):
    print(f"  [{i}] {strip}")

print(f"\n--- Kept strips ({len(res['kept_strips'])}/{len(res['strips'])}) ---")
for i, strip in enumerate(res["kept_strips"], 1):
    print(f"  [{i}] {strip}")

