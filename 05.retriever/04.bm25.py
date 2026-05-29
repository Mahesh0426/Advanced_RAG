from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# BM25 is a keyword-based retriever — no embeddings or vector store required
# It scores documents by term frequency and inverse document frequency (TF-IDF variant)
# TF- Term frequency     IDF - Inverse Doc Freq


# 12 documents spanning medicine, architecture, finance, and literature
docs = [
    Document(page_content="Antibiotics inhibit bacterial cell wall synthesis or protein production to stop infection.", metadata={"topic": "medicine"}),
    Document(page_content="Vaccines introduce antigens to train the immune system to recognise and neutralise pathogens.", metadata={"topic": "medicine"}),
    Document(page_content="MRI scanners use magnetic fields and radio waves to produce detailed images of soft tissue.", metadata={"topic": "medicine"}),
    Document(page_content="Blood pressure is measured in millimetres of mercury (mmHg) and expressed as systolic over diastolic.", metadata={"topic": "medicine"}),
    Document(page_content="The Pantheon in Rome was built around 125 AD and still has the world's largest unreinforced concrete dome.", metadata={"topic": "architecture"}),
    Document(page_content="Gothic cathedrals use flying buttresses to transfer roof weight outward, enabling tall stained glass windows.", metadata={"topic": "architecture"}),
    Document(page_content="The Bauhaus movement combined fine arts and functional design, influencing modern architecture and typography.", metadata={"topic": "architecture"}),
    Document(page_content="Compound interest calculates returns on both the initial principal and previously earned interest.", metadata={"topic": "finance"}),
    Document(page_content="A stock represents partial ownership in a company and entitles the holder to a share of its profits.", metadata={"topic": "finance"}),
    Document(page_content="Diversification reduces portfolio risk by spreading investments across different asset classes.", metadata={"topic": "finance"}),
    Document(page_content="Shakespeare wrote 37 plays and 154 sonnets, exploring themes of power, love, and betrayal.", metadata={"topic": "literature"}),
    Document(page_content="The novel Don Quixote by Cervantes, published in 1605, is often considered the first modern novel.", metadata={"topic": "literature"}),
]

# BM25Retriever builds an inverted index from raw document text
# No embedding model is involved — scoring is purely based on token overlap
retriever = BM25Retriever.from_documents(docs, k=2)


# Query 1: exact keyword match — BM25 excels here
# The words "antibiotic" and "bacterial" appear directly in doc 1
# query1 = "antibiotic bacterial infection treatment and neutralise pathogens"
query1 = "antibiotic bacterial infection treatment and killing pathogens"
results1 = retriever.invoke(query1)

print(f"\nQuery: '{query1}'")
print("\n")

for i, doc in enumerate(results1, 1):
    print(f" [{i}] topic={doc.metadata['topic']}: {doc.page_content}")
print()


# Query 2: keyword match across a different topic
# Words like "compound", "interest", and "returns" are present in finance docs

query2 = "compound interest vs Simple interest and what gives a person partial stakes in a company"
results2 = retriever.invoke(query2)
print(f"\nQuery: '{query2}'")
print("\n")

for i, doc in enumerate(results2, 1):
    print(f"  [{i}] topic={doc.metadata['topic']}: {doc.page_content}")
print()


# Query 3: semantic query with no keyword overlap — BM25 fails here
# None of the docs contain the words "light", "airy", or "tall windows" together
# The Gothic cathedral doc (doc 6) would be the ideal answer, but BM25 likely misses it

query3 = "a structure that feels light and with windows"
print("\n")
results3 = retriever.invoke(query3)
print(f"Query: '{query3}'")
for i, doc in enumerate(results3, 1):
    print(f"  [{i}] topic={doc.metadata['topic']}: {doc.page_content}")