from dotenv import load_dotenv
from typing import Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-5", temperature=0)

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

vectorstore = Chroma.from_documents(docs, 
                                    embedding=embeddings,
                                    collection_name="movies_collection")
                                

# MetadataFilter captures a single filter condition with an optional comparison operator
class MetadataFilter(BaseModel):
    
    field: str = Field(description="The metadata field name to filter on")
    value: str | int | float = Field(description="The value to filter by")
    operator: str = Field(default="eq", description="Comparison operator: eq, ne, gt, gte, lt, lte")
    

# SelfQuerySchema is the structured output the LLM produces from a natural language query
class SelfQuerySchema(BaseModel):
    query: str = Field(description="The semantic search query extracted from the user's query")
    filters: Optional[list[MetadataFilter]] = Field(
        default=None,
        description="Metadata filters extracted from the query. None if no filters apply."
    )

# System prompt tells the LLM what fields are available and how to populate the schema
system_prompt = """You are a query parser for a movie database. Parse the user's natural language query into:
1. A semantic search query (the conceptual meaning to search for)
2. Optional metadata filters on fields: title (string), genre (string), year (integer), rating (float), director (string)

Supported operators: eq, ne, gt, gte, lt, lte
Only add filters when the user explicitly specifies metadata constraints."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "query: {query}"),
])

# with_structured_output binds the Pydantic schema to the LLM — output is always a SelfQuerySchema instance
structured_llm = llm.with_structured_output(SelfQuerySchema)

query_chain = prompt | structured_llm   # pydantic object as output


filters = query_chain.invoke({"query": "What are some movies directed by Christopher Nolan released after 2005"}).filters


class CustomSelfQueryRetriever(BaseRetriever):
    """Retriever that uses an LLM to parse natural language queries into structured filters."""

    vectorstore: Any  # Chroma instance
    query_chain: Any  # prompt | structured_llm chain

    def _build_chroma_filter(self, filters: list[MetadataFilter]) -> dict:
        # Chroma supports operator-based filters: {"field": {"$op": value}}
        # For equality ("eq") the shorthand {"field": value} also works, but we use the full form for consistency
        op_map = {"eq": "$eq", "ne": "$ne", "gt": "$gt", "gte": "$gte", "lt": "$lt", "lte": "$lte"}
        if len(filters) == 1:
            f = filters[0]
            return {f.field: {op_map.get(f.operator, "$eq"): f.value}}  # {'director': {"$eq": "Christopher Nolan"}}
        # Multiple filters are combined with $and
        return {"$and": [{f.field: {op_map.get(f.operator, "$eq"): f.value}} for f in filters]} # {"$and": [{"genre": {"$eq": "sci-fi"}}, {"year": {"$gt": 2005}}]}

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # Step 1: parse the natural language query into a structured schema
        parsed: SelfQuerySchema = self.query_chain.invoke({"query": query})
        print(f"Parsed query: '{parsed.query}'  |  Filters: {parsed.filters}")

        # Step 2: build Chroma-compatible filter if any filters were extracted
        chroma_filter = None
        if parsed.filters:
            chroma_filter = self._build_chroma_filter(parsed.filters)

        # Step 3: run similarity search with the semantic query and optional metadata filter
        return self.vectorstore.similarity_search(parsed.query, k=3, filter=chroma_filter)


retriever = CustomSelfQueryRetriever(vectorstore=vectorstore, query_chain=query_chain)

retriever._build_chroma_filter(filters)

retriever = CustomSelfQueryRetriever(vectorstore=vectorstore, query_chain=query_chain)

# Query 1: genre + year filter
print("=== Sci-fi movies on and after 2005 ===")

results = retriever.invoke("What are some sci-fi movies released on and after 2005?")
for doc in results:
    print(f"  [{doc.metadata['year']}] {doc.metadata['title']} ({doc.metadata['genre']})")
print()

# Query 2: pure semantic — no metadata filter expected

print("=== Movies about dreams ===")
results = retriever.invoke("Movies about dreams and the subconscious mind")
for doc in results:
    print(f"  [{doc.metadata['year']}] {doc.metadata['title']} ({doc.metadata['genre']})")
print()


# Query 3: director equality filter
print("=== Christopher Nolan movies ===")
results = retriever.invoke("What movies did Christopher Nolan direct?")
for doc in results:
    print(f"  [{doc.metadata['year']}] {doc.metadata['title']} - Rating: {doc.metadata['rating']}")