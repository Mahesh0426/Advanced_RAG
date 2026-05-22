from pprint import pp
from langchain_community.document_loaders import JSONLoader
from pathlib import Path

current_dir = Path(__file__).resolve().parent
file_path = current_dir.parent  /"01.document-loaders"/ "knowledge-source" / "apparels.json"
# print("Checking:", file_path)
# print(file_path.exists())

#define loader
loader = JSONLoader(
    file_path=str(file_path),
    jq_schema=".products[]",
    text_content=False,
    # json_lines=True
)

#loads the docs
docs = loader.load(),

pp(f"length of csv: {len(docs)}")
# pp(docs)

#print first json
# pp(docs[0].page_content)

for doc in docs:
      pp(doc.page_content)

  


