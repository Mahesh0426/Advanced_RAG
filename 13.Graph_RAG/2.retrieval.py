import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

load_dotenv()

# temperature=0 keeps Cypher generation deterministic
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
)
# refresh_schema gives the chain an accurate view of node labels and relationship types
graph.refresh_schema()
print(graph.schema)

# Neo4J 5+ requires [:TYPE1|TYPE2|TYPE3] — no colon before subsequent types in a union.
# LLMGraphTransformer title-cases node ids (e.g. "SpaceX" -> "Spacex"), so queries
# must use toLower() to avoid case-mismatch misses.
_CYPHER_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or node labels that are not provided.
Schema:
{schema}

Cypher syntax rules:
1. When matching multiple relationship types with |, only the FIRST type gets a colon prefix:
   Correct:   (n)-[:TYPE1|TYPE2|TYPE3]->(m)
   Incorrect: (n)-[:TYPE1|:TYPE2|:TYPE3]->(m)

2. When a node could have one of several labels, use label union syntax directly in the MATCH clause:
   Correct:   MATCH (n:Organization|Company)
   Incorrect: WHERE (n:`Organization OR n`:Company)

3. Organization and place names are stored with title-cased words. Always compare case-insensitively:
   Correct:   WHERE toLower(n.id) = toLower("SpaceX")
   Incorrect: WHERE n.id = "SpaceX"

4. Person names may be stored in abbreviated or partial forms (e.g. "Musk" instead of "Elon Musk").
   Always match person names with CONTAINS rather than exact equality:
   Correct:   WHERE toLower(p.id) CONTAINS 'elon' AND toLower(p.id) CONTAINS 'musk'
   Incorrect: WHERE toLower(p.id) = toLower("Elon Musk")

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

_cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=_CYPHER_TEMPLATE,
)

# verbose=True prints the generated Cypher so students can see the traversal path
# allow_dangerous_requests is required in langchain-neo4j >= 0.1
cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=_cypher_prompt,
)

# single-hop — one relationship traversal
response = cypher_chain.invoke({"query": "Where is Spacex HQ located?"})
print(response["result"])


# each question requires chaining 2+ relationships — this is what Graph RAG excels at
multi_hop_queries = [
    "What product was released by the organisation whose board Elon Musk left?",
    "At which organisation is Elon Musk's partner Shivon Zilis a director?",
    "In which city is the solar energy company that Tesla acquired headquartered?",
]

for q in multi_hop_queries:
    print(f"\nQ: {q}")
    response = cypher_chain.invoke({"query": q})
    print(f"A: {response['result']}")
    print("-" * 60)