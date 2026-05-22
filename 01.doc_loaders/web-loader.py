from pprint import pp
from langchain_community.document_loaders import WebBaseLoader,RecursiveUrlLoader

url_1 = "https://docs.langchain.com/oss/python/integrations/document_loaders/web_base"
url_2 = "https://docs.langchain.com/oss/python/integrations/document_loaders/pypdfloader"
url_3 = "https://docs.langchain.com/oss/python/integrations/embeddings"


# loader = WebBaseLoader(web_path=[url_1,url_2,url_3])

# docs = loader.load()
# pp(len(docs))
# pp(docs)

base_url = "https://docs.langchain.com/oss/python/integrations/document_loaders"
loader = RecursiveUrlLoader(url=base_url,max_depth=2)

docs = loader.load()
# pp(len(docs))
# pp(docs)

#metadata for 10 docs
for i in range(10):
    doc = docs[i]
    # print(doc.metadata,end="\n\n")

#lazy load
docs_lazy_load = loader.lazy_load()
# pp(docs_lazy_load)

counter = 0
for doc in docs_lazy_load:
    if counter == 20:
        break
    
    counter +=1
    pp(doc.page_content[0:300])
    pp(doc.metadata)


