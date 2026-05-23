# ============================================================
# VECTOR EMBEDDINGS TUTORIAL
# Install: pip install langchain-openai python-dotenv
# Create a .env file with: OPENAI_API_KEY=your_key_here
# ============================================================


from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load your OPENAI_API_KEY from .env file
load_dotenv()


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 1 — What is an Embedding?                         ║
# ║  Text → list of numbers (vector) that captures meaning  ║
# ╚══════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("STEP 1: Create embedder models")
print("="*60)

MODEL_LARGE = "text-embedding-3-large"
MODEL_SMALL = "text-embedding-3-small"

embedder_large = OpenAIEmbeddings(model=MODEL_LARGE)
embedder_small = OpenAIEmbeddings(model=MODEL_SMALL)

print(f"Large model: {MODEL_LARGE}")
print(f"Small model: {MODEL_SMALL}")

# ─────────────────────────────────────────────────────────────
# EXPECTED OUTPUT:
# Large model: text-embedding-3-large
# Small model: text-embedding-3-small


# ─────────────────────────────────────────────────────────────
# LESSON: Two models — large is more accurate but costs more.
#         Small is faster and cheaper but slightly less accurate.
# ─────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 2 — Embed a single query                          ║
# ║  embed_query() → converts ONE string into a vector      ║
# ╚══════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("STEP 2: embed_query() — embed a single sentence")
print("="*60)

query = "What is the Eiffel Tower and where is it located?"

embedding_large = embedder_large.embed_query(query)
embedding_small = embedder_small.embed_query(query)

print(f"\nQuery: '{query}'")
print(f"\nLarge model vector size : {len(embedding_large)}")
print(f"Small model vector size : {len(embedding_small)}")
print(f"\nFirst 5 numbers (large) : {embedding_large[:5]}")
print(f"First 5 numbers (small) : {embedding_small[:5]}")

# ─────────────────────────────────────────────────────────────
# EXPECTED OUTPUT:
# Query: 'What is the Eiffel Tower and where is it located?'
#
# Large model vector size : 3072
# Small model vector size : 1536
#
# First 5 numbers (large) : [0.0023, -0.0041, 0.0198, ...]
# First 5 numbers (small) : [-0.0312, 0.0187, -0.0021, ...]

# ─────────────────────────────────────────────────────────────
# LESSON:
#   large model → 3072 numbers per embedding (more detail)
#   small model → 1536 numbers per embedding (less detail)
#   These numbers capture the MEANING of the sentence.
#   Similar sentences → similar numbers.
# ─────────────────────────────────────────────────────────────




# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 3 — Embed multiple documents                      ║
# ║  embed_documents() → converts a LIST of strings         ║
# ╚══════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("STEP 3: embed_documents() — embed multiple texts at once")
print("="*60)

documents = [
    "The Eiffel Tower is located in Paris, France.",
    "It was built in 1889 and stands 324 meters tall.",
    "The tower was designed by engineer Gustave Eiffel.",
    "It is one of the most visited monuments in the world.",
    "The Eiffel Tower is made of iron and weighs 7,300 tonnes."
]

document_embeddings = embedder_small.embed_documents(texts=documents)

print(f"\nNumber of documents    : {len(documents)}")
print(f"Number of embeddings   : {len(document_embeddings)}")
print(f"Each embedding size    : {len(document_embeddings[0])}")
print(f"\nFirst embedding (doc 1), first 5 numbers:")
print(document_embeddings[0][:5])

# ─────────────────────────────────────────────────────────────
# EXPECTED OUTPUT:
# Number of documents    : 5
# Number of embeddings   : 5
# Each embedding size    : 1536
#
# First embedding (doc 1), first 5 numbers:
# [-0.021, 0.043, -0.018, ...]

# ─────────────────────────────────────────────────────────────
# LESSON:
#   embed_documents() takes a LIST of strings
#   Returns a LIST of vectors — one vector per document
#   5 documents → 5 vectors, each with 1536 numbers
# ─────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 4 — embed_query vs embed_documents                ║
# ║  Why are there TWO methods?                             ║
# ╚══════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("STEP 4: embed_query() vs embed_documents() — the difference")
print("="*60)

print("""
embed_query()       → for the USER's search question
embed_documents()   → for the DOCUMENTS you want to search through

Real world flow:
  
  User types:  "How tall is Eiffel Tower?"
                        │
                        ▼
               embed_query()           ← embed the question
                        │
                        ▼
            [0.021, -0.043, 0.018...]  ← question vector
                        │
                        ▼
         Compare against document vectors
         (find the most similar one)
                        │
                        ▼
         Return: "It stands 324 meters tall." ✅
""")


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 5 — Chunk text THEN embed                         ║
# ║  Real RAG flow: long text → chunks → embeddings         ║
# ╚══════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("STEP 5: Chunk text first, then embed each chunk")
print("="*60)

long_text = """The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.

The tower is 330 metres tall and is the tallest structure in Paris.
Its base is square, measuring 125 metres on each side.
During its construction, the Eiffel Tower surpassed the Washington Monument to become the world's tallest man-made structure.

The tower has three levels for visitors, with restaurants on the first and second levels.
The top level's upper platform is the highest observation deck accessible to the public in the European Union.
Tickets can be purchased to ascend by stairs or lift to the first and second levels.

The tower receives around 7 million visitors annually.
It was initially criticised by some of France's leading artists and intellectuals for its design.
But it has become a global cultural icon of France and one of the most recognisable structures in the world."""

# Step 5a: Chunk the text
chunker = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
chunks = chunker.split_text(long_text)

print(f"\nOriginal text length : {len(long_text)} chars")
print(f"Number of chunks     : {len(chunks)}")
for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i} ({len(chunk)} chars):\n{chunk}")

# Step 5b: Embed each chunk
print("\n" + "-"*40)
print("Embedding all chunks...")
chunk_embeddings = embedder_small.embed_documents(texts=chunks)

print(f"\nNumber of chunk embeddings : {len(chunk_embeddings)}")
print(f"Each embedding size        : {len(chunk_embeddings[0])}")

# ─────────────────────────────────────────────────────────────
# EXPECTED OUTPUT:
# Original text length : 931 chars
# Number of chunks     : 6
#
# Chunk 1 (198 chars):
# The Eiffel Tower is a wrought-iron lattice tower...
#
# Chunk 2 (185 chars):
# The tower is 330 metres tall...
# ...
#
# Number of chunk embeddings : 6
# Each embedding size        : 1536
# ─────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 6 — Similarity: do similar texts have             ║
# ║  similar vectors? (cosine similarity)                   ║
# ╚══════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("STEP 6: Similar text = similar vectors (cosine similarity)")
print("="*60)

import numpy as np

def cosine_similarity(vec1, vec2):
    """1.0 = identical meaning, 0.0 = completely different"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

sentence_a = "The Eiffel Tower is in Paris"
sentence_b = "The famous Paris tower was built by Gustave Eiffel"   # similar meaning
sentence_c = "Python is a popular programming language"              # different meaning

emb_a = embedder_small.embed_query(sentence_a)
emb_b = embedder_small.embed_query(sentence_b)
emb_c = embedder_small.embed_query(sentence_c)

sim_ab = cosine_similarity(emb_a, emb_b)
sim_ac = cosine_similarity(emb_a, emb_c)

print(f"\nSentence A : '{sentence_a}'")
print(f"Sentence B : '{sentence_b}'")
print(f"Sentence C : '{sentence_c}'")
print(f"\nSimilarity A vs B (same topic)    : {sim_ab:.4f}")
print(f"Similarity A vs C (diff topic)    : {sim_ac:.4f}")
print(f"\nConclusion: A and B are {'MORE' if sim_ab > sim_ac else 'LESS'} similar ✅")

# ─────────────────────────────────────────────────────────────
# EXPECTED OUTPUT:
# Similarity A vs B (same topic)    : 0.8921   ← HIGH (similar meaning)
# Similarity A vs C (diff topic)    : 0.2134   ← LOW  (different meaning)
#
# Conclusion: A and B are MORE similar ✅
# ─────────────────────────────────────────────────────────────
# LESSON: This is exactly how RAG finds relevant chunks!
#         Query vector vs all document vectors → pick highest similarity
# ─────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════╗
# ║  SUMMARY                                                ║
# ╚══════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("""
Step 1 → Two models: large (3072 dims) vs small (1536 dims)
Step 2 → embed_query()     → single string  → one vector
Step 3 → embed_documents() → list of strings → list of vectors
Step 4 → query vs documents: two different methods, one purpose
Step 5 → Real flow: long text → chunk → embed each chunk
Step 6 → Similar meaning = similar vectors (cosine similarity)

Full RAG Pipeline:
  Documents → chunk → embed → store in vector DB
  User query → embed → compare → return similar chunks → LLM
""")