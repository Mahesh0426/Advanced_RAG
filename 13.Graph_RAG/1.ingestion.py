import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, Neo4jVector

load_dotenv()

llm = ChatOpenAI(model = "gpt-5-mini",temperature = 0)  
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")

# PyPDFLoader yields one Document per page
loader = PyPDFLoader("data/elon_musk.pdf")
pages = loader.load()

for i, p in enumerate(pages):
    print(f"Page {i + 1}: {len(p.page_content)} chars")
    


# smaller chunks give the LLM tighter context for entity extraction
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(pages)

print(f"{len(chunks)} chunks created")

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
)


graph_transformer = LLMGraphTransformer(llm=llm)


graph_docs = graph_transformer.convert_to_graph_documents(chunks)

print(f"{len(graph_docs)} graph documents extracted")

# spot-check the first extraction
print("Nodes:", [n.id for n in graph_docs[0].nodes])
print("Rels: ", [(r.source.id, r.type, r.target.id) for r in graph_docs[0].relationships])



# include_source=True links each entity node back to its source Document node,
# which is required for Neo4jVector.from_existing_graph in the next cell
graph.add_graph_documents(
    graph_docs,
    include_source=True,
    baseEntityLabel=True
)
print("Graph stored in Neo4J")



# create a vector index over the Document nodes stored above
vector_index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    index_name="elon_musk_chunks",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)
print("Vector index created")




# verify what landed in Neo4J
node_counts = graph.query(
    "MATCH (n) RETURN labels(n) AS label, count(n) AS count ORDER BY count DESC"
)
rel_counts = graph.query(
    "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC"
)
print("Nodes:")
for r in node_counts:
    print(" ", r)
print("Relationships:")
for r in rel_counts:
    print(" ", r)