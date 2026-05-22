from pprint import pp
from langchain_community.document_loaders import TextLoader
from pathlib import Path


#define the path of the text file
# file_path = Path(__file__).parent / "knowledge-source" / "transformers.txt"


current_dir = Path(__file__).resolve().parent
file_path = current_dir.parent  /"01.document-loaders"/ "knowledge-source" / "transformers.txt"
# print("Checking:", file_path)
# print(file_path.exists())

#define loader
loader = TextLoader(file_path=file_path, encoding="windows-1252")

#loads the docs
docs = loader.load()
# print(type(docs))   
# print(type(docs[0]))
# print(docs)

#text of the documenst
extracted_docs = docs[0]
# print(extracted_docs.page_content)

#metadata
print(extracted_docs.metadata)

#length of docs 
print(len( "length:",docs))