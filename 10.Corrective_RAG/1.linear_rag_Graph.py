from typing import List, TypedDict
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
    question: str # the user's input query
    docs: List[Document] #the retrieved chunks from FAISS
    answer: str # the final LLM response

# 6. Retrieve node - 1st NODE. It takes the question from the state and returns the retrieved chunks.
def retrieve(state):
    print("\n--- retrieve node ---")
    q=state["question"]
    print("Question:", q)
    docs = retriever.invoke(q)
    print(f"Retrieved {len(docs)} chunks\n")
    return {"docs": docs}

# 7. Prompt template + generate node
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the context. if not in context, say you don't know.."),
    ("human"," Question: {question}\n\nContext: \n{context}")
])

# 8. Generate node - 2nd NODE. It takes the retrieved chunks from the state and returns the final answer.
def generate(state):
    print("\n--- generate node ---")
    context = "\n\n".join([d.page_content for d in state["docs"]])
    # print("Context:\n", context)
    out = (prompt | llm).invoke({"question":state["question"],"context":context}) 
    return {"answer" : out.content}


# 9. build the RAG graph - START → retrieve → generate → END
g = StateGraph(State)
g.add_node("retrieve",retrieve)
g.add_node("generate",generate)
g.add_edge(START,"retrieve")
g.add_edge("retrieve","generate")
g.add_edge("generate",END)

# 10. compile the graph into a runnable pipeline
workflow = g.compile()

#RUN
res = workflow.invoke(
    {"question": "explain how Corrective rag works?", "docs":[],"answer":""} 
    # {"question": "what is deep learning?", "docs":[],"answer":""} 
)

print("\n--- Final answer: ---")
print(res["answer"])

# print("="*100)
# print(res["docs"][0].page_content)
# print("*"*100)
# print(res["docs"][1].page_content)
# print("*"*100)
# print(res["docs"][2].page_content)
# print("*"*100)
# print(res["docs"][3].page_content)




    