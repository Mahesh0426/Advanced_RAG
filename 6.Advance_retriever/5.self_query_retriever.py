from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_community.query_constructors.chroma import ChromaTranslator

# Load API keys and environment variables from .env file
load_dotenv()

# Initialize the embedding model to convert text into vector representations.
# "text-embedding-3-small" is OpenAI's efficient model for semantic similarity tasks.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize the LLM that will parse natural language queries into structured filters.
# temperature=0 ensures deterministic, consistent query parsing (no randomness).
llm = ChatOpenAI(model="gpt-5", temperature=0)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET: A small in-memory movie corpus.
# Each Document has:
#   - page_content : free-text plot description (used for semantic/vector search)
#   - metadata     : structured fields (used for hard filters like genre, year, rating)
# ─────────────────────────────────────────────────────────────────────────────
docs = [
    Document(
        page_content="A masked vigilante fights crime in a corrupt city with the help of a billionaire's technology. An iconic supervillain pushes him to his limits in a battle for Gotham's soul.",
        metadata={"title": "The Dark Knight", "genre": "action", "year": 2008, "rating": 9.0, "director": "Christopher Nolan"},
    ),
    Document(
        page_content="A thief who steals secrets through dream-sharing technology is offered a chance to have his past erased if he can plant an idea in someone's mind. A visually stunning exploration of the subconscious.",
        metadata={"title": "Inception", "genre": "sci-fi", "year": 2010, "rating": 8.8, "director": "Christopher Nolan"},
    ),
    Document(
        page_content="A team of explorers travels through a wormhole in space to find a new habitable planet for humanity. Stunning visuals of black holes and time dilation challenge our understanding of physics.",
        metadata={"title": "Interstellar", "genre": "sci-fi", "year": 2014, "rating": 8.6, "director": "Christopher Nolan"},
    ),
    Document(
        page_content="A programmer discovers that reality is a simulation and joins a rebellion against the machines controlling humanity. A groundbreaking blend of philosophy, martial arts, and bullet-time action.",
        metadata={"title": "The Matrix", "genre": "sci-fi", "year": 1999, "rating": 8.7, "director": "Lana Wachowski"},
    ),
    Document(
        page_content="Two criminals and a mob boss's wife are caught in a web of violence and dark humor over a single eventful day in Los Angeles. Interweaving storylines told out of chronological order.",
        metadata={"title": "Pulp Fiction", "genre": "drama", "year": 1994, "rating": 8.9, "director": "Quentin Tarantino"},
    ),
    Document(
        page_content="A maverick surgeon navigates the chaotic social landscape of a mobile army unit during the Korean War. Sharp satirical comedy disguised as a war film, later adapted into a beloved TV series.",
        metadata={"title": "MASH", "genre": "comedy", "year": 1970, "rating": 7.4, "director": "Robert Altman"},
    ),
    Document(
        page_content="Humanity sends a last-ditch mission to reignite the dying sun with a massive stellar bomb. An intense psychological thriller set in the terrifying emptiness of deep space.",
        metadata={"title": "Sunshine", "genre": "sci-fi", "year": 2007, "rating": 7.3, "director": "Danny Boyle"},
    ),
    Document(
        page_content="A soldier wakes up in another man's body aboard a commuter train just minutes before it explodes, reliving the event repeatedly to identify the bomber. A clever sci-fi thriller about time loops and identity.",
        metadata={"title": "Source Code", "genre": "sci-fi", "year": 2011, "rating": 7.5, "director": "Duncan Jones"},
    ),
]

print(f"Created {len(docs)} movie documents")

# ─────────────────────────────────────────────────────────────────────────────
# VECTOR STORE: Embed all documents and store them in a local ChromaDB collection.
# Each document's page_content is converted into a vector and indexed.
# This enables semantic similarity search later.
# ─────────────────────────────────────────────────────────────────────────────
vectorstore = Chroma.from_documents(docs, 
                                    embedding=embeddings,
                                    collection_name="movies_collection")

print("\n Vector store is ready..\n")

# ─────────────────────────────────────────────────────────────────────────────
# METADATA SCHEMA: Describe every filterable metadata field to the LLM.
# The LLM uses these descriptions to understand what fields it can filter on
# when it parses a natural language query into a structured query.
#
# Example: For "sci-fi movies after 2010", the LLM knows:
#   - "genre" is a string it can match against → genre == "sci-fi"
#   - "year" is an integer it can compare → year > 2010
# ─────────────────────────────────────────────────────────────────────────────
metadata_field_info = [
    AttributeInfo(name="title", description="The title of the movie", type="string"),
    AttributeInfo(name="genre", description="The genre of the movie (action, sci-fi, drama, comedy)", type="string"),
    AttributeInfo(name="year", description="The year the movie was released", type="integer"),
    AttributeInfo(name="rating", description="The IMDb rating of the movie (0-10)", type="float"),
    AttributeInfo(name="director", description="The director of the movie", type="string"),
]

# A plain-English description of what the page_content field contains.
# The LLM uses this to understand which part of the query should drive
# semantic search vs. which part should become a metadata filter.
document_content_description = "Brief plot descriptions of movies"

# ─────────────────────────────────────────────────────────────────────────────
# SELF QUERY RETRIEVER: The core component of this script.
#
# How it works (two-step process):
#   Step 1 — Query Construction (LLM):
#     The LLM reads the natural language query and splits it into:
#       a) A semantic search string  → searches against page_content vectors
#       b) A structured metadata filter → applied as a hard WHERE-style clause
#
#   Step 2 — Filtered Vector Search (ChromaDB):
#     ChromaDB runs the vector similarity search with the metadata filter applied,
#     so only documents that pass both conditions are returned.
#
# Example — query: "sci-fi movies released after 2010"
#   → semantic query : "sci-fi movies"
#   → metadata filter: { genre == "sci-fi" AND year > 2010 }
#
# ChromaTranslator converts the LLM's abstract filter AST into ChromaDB's
# native "where" clause format that Chroma can actually execute.
#
# enable_limit=True allows the LLM to also extract a result count from the query.
# Example — "recommend me 2 sci-fi movies" → limit=2 is passed to the search.
# ─────────────────────────────────────────────────────────────────────────────
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    structured_query_translator=ChromaTranslator(),
    enable_limit=True
)

# ─────────────────────────────────────────────────────────────────────────────
# TEST QUERIES — uncomment to explore different retrieval behaviours:
#
#   "What are some sci-fi movies released after 2010"
#     → filter: genre=="sci-fi" AND year>2010
#
#   "Recommend me 2 sci-fi movies released after 2000"
#     → filter: genre=="sci-fi" AND year>2000, limit=2
#
#   "movie about a superhero who is a billionaire by day and a masked vigilante by night"
#     → pure semantic search (no metadata filter), matches The Dark Knight by plot similarity
#
#   "What movies did Christopher Nolan direct?"
#     → filter: director=="Christopher Nolan" (no semantic component needed)
# ─────────────────────────────────────────────────────────────────────────────
# query = "What are some sci-fi movies released after 2010"
# query = "Recommend me 2 sci-fi movies released after 2000"
# query = "movie about a superhero who is a billionaire by day and a masked vigilante by night"
query = "What movies did Christopher Nolan direct?"


print("\nquery:",query)
print()

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE: Plain vector similarity search (no metadata filtering).
# Converts the query into a vector and returns the top-k most similar documents
# purely based on semantic closeness of their plot descriptions.
#
# Limitation: For a query like "Christopher Nolan movies", this may return
# thematically similar films from other directors because it only understands
# meaning, not structured facts like who directed what.
# ─────────────────────────────────────────────────────────────────────────────
vs_retriever = vectorstore.as_retriever(search_type="similarity",
                                        search_kwargs={"k": 3})
results = vs_retriever.invoke(query)
print("\nSimalarity Search:\n")
for doc in results:
    print(f"[{doc.metadata['year']}] {doc.metadata['title']} ({doc.metadata['genre']}) - dir. {doc.metadata['director']}")
    print(f"  {doc.page_content[:100]}...")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SELF QUERY RETRIEVAL: Combines semantic search with LLM-generated metadata filters.
#
# For query "What movies did Christopher Nolan direct?":
#   The LLM extracts → filter: { director == "Christopher Nolan" }
#   ChromaDB applies the filter first, then ranks results by vector similarity.
#
# This guarantees that ONLY Christopher Nolan films are returned,
# unlike plain similarity search which might surface other directors' films.
# ─────────────────────────────────────────────────────────────────────────────
results = retriever.invoke(query) # metadata --> {"genre": "sci-fi", "year": >= 2005}
print("\nSelfQueryRetriever:\n")
for doc in results:
    print(f"[{doc.metadata['year']}] {doc.metadata['title']} ({doc.metadata['genre']}) - dir. {doc.metadata['director']}")
    print(f"  {doc.page_content[:100]}...")
    print()