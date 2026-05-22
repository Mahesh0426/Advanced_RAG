from pprint import pp
from langchain_community.document_loaders import CSVLoader
from pathlib import Path

current_dir = Path(__file__).resolve().parent
file_path = current_dir.parent  /"01.document-loaders"/ "knowledge-source" / "organizations.csv"
# print("Checking:", file_path)
# print(file_path.exists())

#define loader
loader = CSVLoader(
    file_path=file_path,
    # source_column="Industry",
    # metadata_columns = ["Website","Founded","Number of employees"],
    # content_columns = ["Description"]
    
    )

#loads the docs
docs = loader.load()
pp(f"length of csv: {len(docs)}")

# first five column
# pp(docs[0:5])

#first row
pp(docs[0].page_content)

pp(f"metadata: {docs[0].metadata}" )
