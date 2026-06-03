### RAG Fusion Pipeline - Ensemble Retriever
#  =============================================================================
# This notebook demonstrates RAG Fusion using **two retrievers combined via an Ensemble Retriever**.
# Instead of generating sub-queries with an LLM, we use two different retrieval strategies on the same vector store and fuse their results using **Reciprocal Rank Fusion (RRF)** built into LangChain's `EnsembleRetriever`.

# **Retrievers used:**
# - **Similarity Search** - standard cosine similarity retrieval
# - **MMR (Maximal Marginal Relevance)** - balances relevance with diversity (`lambda_mult=0.5`)

# **Steps covered:**
# 1. Load the source PDF
# 2. Split documents into chunks
# 3. Generate embeddings and store in ChromaDB
# 4. Create two retrievers (Similarity + MMR)
# 5. Apply RAG Fusion via Ensemble Retriever with weights [0.5, 0.5]
# 6. Augmentation - build context from fused documents
# 7. Generation - produce a grounded answer using an LLM

# =============================================================================


import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from rag_fusion import RAGFusion

# Load OPENAI_API_KEY from the .env file
load_dotenv()


### ----------- STEP 1 - Load the PDF -----------
# PyPDFLoader` reads the PDF and returns one `Document` object per page.

base_dir = os.path.dirname(__file__)
pdf_path = os.path.join(base_dir, "notebooklm_rag.pdf")

loader = PyPDFLoader(pdf_path)
pages = loader.load()

print(f"Loaded {len(pages)} page(s) from the PDF.")

### -----------STEP 2 - Split Documents into Chunks -------------
# Large pages are split into smaller, overlapping chunks so that the retriever can surface focused, relevant passages rather than entire pages.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)
print(f"\nSplit into {len(chunks)} chunk(s).\n")

### ------------STEP 3 - Embeddings & Vector Store-------
# Each chunk is converted into a dense vector using OpenAI's `text-embedding-3-small` model and stored in a ChromaDB vector store.
# Both retrievers will query this same vector store using different strategies.
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="notebooklm_rag"
)
print("\nVector store created successfully.")

### -----------STEP 4 - Create Two Retrievers-----------
# We configure a similarity-search retriever with `k=3`, meaning it will return the 3 most relevant chunks for any given query.
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

### -----------STEP 5 - RAG Fusion with Ensemble Retriever------------
# RAGFusion.from_llm` wires up the LLM to generate multiple sub-queries from the original query.
# Each sub-query is sent to the retriever independently, and the results are merged using **Reciprocal Rank Fusion (RRF)** - documents that rank highly across multiple sub-queries bubble to the top.
llm = ChatOpenAI(model="gpt-4o-mini")
# Build the RAG Fusion pipeline: LLM generates 2 sub-queries, retrieves docs for each, then fuses
rag_fusion = RAGFusion.from_llm(
    llm=llm,
    retriever=retriever,
    num_subqueries=2,
    k=3
)

### ----------- STEP 6 Querying -----------
query = "How does NotebookLM retrieve relevant information from uploaded documents?"
print("\nQuery:", query)


# Both retrievers run in parallel; EnsembleRetriever fuses the results via RRF
fused_docs = rag_fusion.invoke(query)

print(f"\nRetrieved {len(fused_docs)} fused document(s).")
for i, doc in enumerate(fused_docs):
    print(f"\n--- Document {i + 1} ---")
    print(doc.page_content)
    
#  ---------Step 7 Augmentation -----------
#  The retrieved chunks are concatenated into a single context string.
# This context will be injected into the generation prompt to ground the LLM's answer.
# Join all retrieved chunks into one context block

context = "\n\n".join([doc.page_content for doc in fused_docs])
print("\nCONTEXT:")
print(context)
print("---------------------------------")

# -----------STEP 8 Generation------------
# The context and original query are passed to the LLM via a structured prompt.
# The LLM is instructed to answer **only** from the provided context and to say `"I don't know"` if the answer isn't there.

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the context provided below to answer the question.
Be clear, concise, and accurate in your response.
If the answer is not present in the context, say "I don't know" - do not make up an answer.

Context:
{context}

Query: {query}

Answer:
""")

# Chain: prompt -> LLM
generation_chain = prompt | llm
response = generation_chain.invoke({"context": context, "query": query})
print("\n\nLLM ANSWER:")
print(response.content)