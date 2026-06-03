# ### RAG Fusion Pipeline - Ensemble Retriever

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
#----------------------------------------------------



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
    collection_name="notebooklm_rag_ensemble"
)
print("\nVector store created successfully.")

### -----------STEP 4 - Create Two Retrievers-----------
# We create two retrievers from the same vector store, each using a different search strategy:
# - **Similarity Search** - ranks chunks purely by cosine similarity to the query
# - **MMR** - Maximal Marginal Relevance reduces redundancy by penalizing chunks that are too similar to already-selected ones. `lambda_mult=0.5` balances relevance and diversity equally.
# standard cosine similarity search

# Retriever 1: standard cosine similarity search
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Retriever 2: MMR promotes diverse results alongside relevant ones
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

# ### -----------STEP 5 - RAG Fusion with Ensemble Retriever------------
# RAGFusion.from_retrievers` wraps both retrievers inside LangChain's `EnsembleRetriever`.
# Equal weights `[0.5, 0.5]` mean both retrievers contribute equally to the final RRF ranking.
# When `invoke` is called, the ensemble retriever queries both strategies and merges the results.
rag_fusion = RAGFusion.from_retrievers(
    base_retrievers=[similarity_retriever, mmr_retriever],
    weights=[0.5, 0.5],
    k=3
)
print("\nRAG Fusion pipeline created successfully using Ensemble Retriever.")


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
llm = ChatOpenAI(model="gpt-4o-mini")
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