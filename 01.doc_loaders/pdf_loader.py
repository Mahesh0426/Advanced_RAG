from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import PDFMinerLoader, PDFPlumberLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser

from pathlib import Path
from pprint import pp

#create the path for pdf 
current_dir = Path(__file__).resolve().parent
pdf_path = current_dir.parent  /"01.document-loaders"/ "knowledge-source" / "attention_is_all_you_need.pdf"
# print(file_path.exists())


#1. create a loader to load pdf
pdf_loader = PyPDFLoader(file_path=str(pdf_path))
docs = pdf_loader.load()
# pp(docs)
pp(f"length: {len(docs)}")


#fetch the first page text content
# pp(docs[0].page_content)

#metadata
# pp(docs[0].metadata)

#2. create instance which can etract images
Pdfminer_loader = PDFMinerLoader(
    file_path=str(pdf_path), 
    mode = "page",
    extract_images=True,
    images_parser=RapidOCRBlobParser(),
    images_inner_format="html-img"
    
)

#load doc with imgs
image_loader = Pdfminer_loader.load()
# pp(f"image loader:\n{image_loader[2].page_content[-450:]}")


# 3.text from table
# pp(image_loader[5].page_content)

#4. PDF plumber | to get detailed  metadata from pdf
plumbeer_loader = PDFPlumberLoader(file_path=str(pdf_path))
docs_with_metadata = plumbeer_loader.load()
pp(docs_with_metadata[0].metadata)






