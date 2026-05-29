## Did text come from a Loader?

When working with LangChain text processing, the way you split data depends on where the text comes from.

---

### Did text come from a Loader?

### YES → `split_documents()`

Use this when your data comes from document loaders like:

- PDF loaders
- Web page loaders
- Text file loaders
- Database loaders

Each item is already a `Document` object with metadata.

This is the correct approach for real-world RAG pipelines.

**Why use it:**

- Preserves metadata (source, page number, etc.)
- Works directly with `Document` objects
- Designed for production pipelines

---

### NO → `split_text()`

Use this when you only have raw text (a plain string), not documents.

This is fine for:

- Learning
- Quick experiments
- Simple prototypes

**Why use it:**

- Lightweight
- No need for document structure
- Fast for testing ideas

---

### Quick rule of thumb

- Loader output → `split_documents()`
- Raw string → `split_text()`

## Similarity Metrics in Vector Search (RAG, Embeddings)

When working with embeddings (like in RAG systems), we need a way to measure how “close” two vectors are. This is called a similarity or distance metric.

Different metrics behave differently, and choosing the right one matters for search quality.

---

### 1. Cosine Similarity

Cosine similarity measures the **angle between two vectors**, not their actual length.

### Formula:

cosine similarity = (A · B) / (||A|| \* ||B||)

### Key idea:

It checks whether two texts point in the same direction in embedding space.

### Range:

- `1` → identical meaning
- `0` → unrelated
- `-1` → opposite meaning

### When to use:

- Most common in NLP and RAG
- Works well when text length varies

---

## 2. Cosine Distance

Cosine distance is simply the inverse of cosine similarity.

### Formula:

cosine distance = 1 - cosine similarity

### Key idea:

- Smaller value = more similar
- Larger value = less similar

### When to use:

- When a “distance” score is required instead of similarity

---

## 3. Euclidean Distance (L2 Distance)

Euclidean distance measures the **straight-line distance between two points** in space.

### Formula:

d(A, B) = √(Σ (Ai - Bi)²)

### Key idea:

- Measures actual geometric distance
- Sensitive to vector magnitude

### When to use:

- When absolute position matters
- Image embeddings, clustering problems

### Limitation:

- Not ideal for text embeddings (because length affects results)

---

## 4. Manhattan Distance (L1 Distance)

Manhattan distance measures distance by moving along grid lines (like city blocks).

### Formula:

d(A, B) = Σ |Ai - Bi|

### Key idea:

- Adds absolute differences across dimensions
- Less sensitive to outliers than Euclidean

### When to use:

- Sparse data
- Some ML clustering problems

---

## 5. Dot Product

Dot product measures how aligned two vectors are, including magnitude.

### Formula:

A · B = Σ (Ai \* Bi)

### Key idea:

- Higher value = more similar
- Considers both direction and magnitude

### When to use:

- When vector magnitude is meaningful
- Some embedding models are trained for dot product similarity

---

## Quick Comparison

| Metric             | What it measures       | Best for               |
| ------------------ | ---------------------- | ---------------------- |
| Cosine similarity  | Angle between vectors  | NLP, RAG, embeddings   |
| Cosine distance    | 1 - similarity         | Distance-based systems |
| Euclidean distance | Straight-line distance | Images, clustering     |
| Manhattan distance | Grid path distance     | Sparse data, ML tasks  |
| Dot product        | Alignment + magnitude  | Some embedding models  |

---

## Simple intuition

- Cosine similarity → “Do these mean the same thing?”
- Euclidean → “How far apart are they physically?”
- Manhattan → “How many steps to get there?”
- Dot product → “How strongly do they align?”

---

# HNSW (Hierarchical Navigable Small World) in Vector Databases

HNSW is one of the most important algorithms used for **fast similarity search in vector databases** like Chroma, Pinecone, Weaviate, and FAISS (approx variants).

It helps find nearest vectors quickly, even when you have millions or billions of embeddings.

---

## What problem does HNSW solve?

When you store embeddings, a simple search would compare your query vector with **every vector in the database**.

That is called **brute force search**:

- Accurate but very slow
- Not scalable for large datasets

HNSW solves this by making search **approximate but extremely fast**.

---

## Core idea of HNSW

HNSW builds a **graph of vectors**, where:

- Each vector is a node
- Similar vectors are connected with edges

Instead of checking every vector, it:

- Starts from a random or top-level node
- “Navigates” through neighbors
- Moves closer step by step to the nearest match

Think of it like Google Maps navigation instead of checking every possible route.

---

# chromaDB

### Choosing Between `from_documents()` and `Chroma() + add_documents()`

When working with Chroma, there are two main ways to create your vector store. Both are valid, but each is suited for different scenarios.

---

### 1. When to use `from_documents()`

`from_documents()` is a simple helper that embeds your documents and builds the vector store in a single step.

Use this approach if:

- You are testing
- You are building quick demos
- You don’t need persistence

**Example:**

```python
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="demo_collection",
)
```

### 2. When to use `Chroma(...) + add_documents()`

This approach gives you full control over how the database is created, updated, and persisted. It works well in real applications.

Use this approach if:

- You want persistence
- You plan to reload the DB later
- You want to manage the DB lifecycle
- You want a clean backend architecture

**Example:**

```python
vector_store = Chroma(
    collection_name="demo_1",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

vector_store.add_documents(documents)
```
