from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_community.query_constructors.chroma import ChromaTranslator

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

print("\n Vector store is ready..\n")

# AttributeInfo tells the LLM what metadata fields exist and how to filter on them
metadata_field_info = [
    AttributeInfo(name="title", description="The title of the movie", type="string"),
    AttributeInfo(name="genre", description="The genre of the movie (action, sci-fi, drama, comedy)", type="string"),
    AttributeInfo(name="year", description="The year the movie was released", type="integer"),
    AttributeInfo(name="rating", description="The IMDb rating of the movie (0-10)", type="float"),
    AttributeInfo(name="director", description="The director of the movie", type="string"),
]


document_content_description = "Brief plot descriptions of movies"

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    structured_query_translator=ChromaTranslator(),
    enable_limit=True
)

# query = "What are some sci-fi movies released after 2010"
# query = "Recommend me 2 sci-fi movies released after 2000"
# query = "movie about a superhero who is a billionaire by day and a masked vigilante by night"
query = "What movies did Christopher Nolan direct?"


print("\nquery:",query)
print()

# create the vs retriever
vs_retriever = vectorstore.as_retriever(search_type="similarity",
                                        search_kwargs={"k": 3})
results = vs_retriever.invoke(query)
print("\nSimalarity Search:\n")
for doc in results:
    print(f"[{doc.metadata['year']}] {doc.metadata['title']} ({doc.metadata['genre']}) - dir. {doc.metadata['director']}")
    print(f"  {doc.page_content[:100]}...")
    print()

# LLM extracts: semantic query="sci-fi movies", filter={genre: "sci-fi", year > 2005}
results = retriever.invoke(query) # metadata --> {"genre": "sci-fi", "year": >= 2005}
print("\nSelfQueryRetriever:\n")
for doc in results:
    print(f"[{doc.metadata['year']}] {doc.metadata['title']} ({doc.metadata['genre']}) - dir. {doc.metadata['director']}")
    print(f"  {doc.page_content[:100]}...")
    print()