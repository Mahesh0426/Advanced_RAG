from typing import List, TypedDict,Literal
from pydantic import BaseModel
import re
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
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

UPPER_TH = 0.7
LOWER_TH = 0.3

class State(TypedDict):
    quetion: str
    docs: List[Document]

    good_docs: List[Document]
    verdict:str
    reason:str

    strips: List[str]
    kept_strips: List[str]
    refied_strips: List[str]

    answer:str

def retrieve_node(state: State) -> State:
    q = state["quetion"]
    return {"docs": retriever.invoke(q)}

#score based doc evaluator
class DocEvaluator(BaseModel):
    score: float
    reason: str

# system prompt
doc_eval_prompt = ChatPromptTemplate.from_messages([(
    "system", 
    "You are a strict retrieval evaluator for RAG.\n"
    "you will be given ONE retrieved chunks and a question.\n"
    "return a relevance score in [0.0 ,1.0]\n"
    "- 1.0: chunk alone is sufficient to answer fully/mostly.\n"
    "- 0.0: chunk is completely irrelevant.\n"
    "Be conservative with high scores\n"
    "Also return  a short reason\n"
    "output JSON only"

),
("Human","Question: {quetion}\n\nChunk: {chunk}\n\n")
])

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvaluator)

def eval_each_doc_node(state: State) -> State:
    q = state["quetion"]
    scores: List[float] = []
    reasons: List[str] = []
    good: List[Document] = []

    for d in state["docs"]:
        out = doc_eval_chain.invoke({"quetion": q, "chunk": d.page_content})
        scores.append(out.score)
        reasons.append(out.reason)

       # for INCORRECT case we will refine docs with score > LOWER_TH
        if out.score > LOWER_TH:
            good.append(d)
        
        #CORRECT if at least one doc has score > UPPER_TH
        if any(s > UPPER_TH for s in scores):
            return{
                "good_docs": good,
                "verdict": "CORRECT",
                "reason": "At least one chunk score < {LOWER_TH}.{why}"
            }
        if len(scores) > 0 and all(s < LOWER_TH for s in scores):
            why = "No chunk was sifficient"
            return {
                "good_docs": [],
                "verdict": "INCORRECT",
                "reason": f"All  retrieved chunks scored < {LOWER_TH}. {why}"
            }
        
        # anything in between => AMBIGUOUS
        why = "Mixed relavance signals."
        return {
            "good_docs": good,
            "verdict": "AMBIGUOUS",
            "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}. {why}"
        }

# sentence level DECOMPOSER
def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r'\s+', " ", text).strip() # normalize whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20] # filter out long sentences

#filter LLM as Judge
class KeepOrDrop(BaseModel):
    keep:bool

filter_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are a strict relevance filter.\n"
        "Return keep=true if the sentence directly helps answer the quetion.\n"
        "Use ONLY the sencences. Output JSON only."
    ),
    ("human",
    "Question: {quetion}\n\nSentence: {sentence}\n\n"
)
      ])

# for each good doc, we will decompose it into sentences and filter them with the LLM judge    
filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)

# refining decompose -> filter -> recompose

def refine(state: State) -> State:
    q = state["question"]

    #combine retrieved docs into one context string
    # In correct case , evall node populates good_docs with docs having scores > LOWER_Th
    context = "\n\n".join(d.page_content for d in state["good_docs"]).strips()
    



