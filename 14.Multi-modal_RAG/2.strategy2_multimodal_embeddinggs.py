"""
Strategy 2 – Multimodal RAG using CLIP Embeddings
===================================================
Instead of converting images to text captions (Strategy 1), this approach:
  1. Embeds BOTH text chunks AND raw images into the SAME vector store
     using OpenCLIP (a vision-language model that creates a shared embedding
     space for text and images).
  2. At query time, the retriever can return relevant text chunks AND images.
  3. The matched images are base64-encoded and sent directly to a
     vision-capable LLM (GPT-4o-mini) alongside the text context.

Key difference from Strategy 1:
  - No captioning step → images are embedded as visual features, not text.
  - Requires OpenCLIP (open_clip) and a Chroma vector store that supports
    image URIs via add_images().
"""

import os
import base64                        # for encoding images to base64 strings
from dotenv import load_dotenv       # reads .env file (OPENAI_API_KEY, etc.)
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain_openai import ChatOpenAI
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage   # wraps multimodal content

# ── Environment setup ─────────────────────────────────────────────────────────
load_dotenv()   

import open_clip
# Uncomment to see all available CLIP model/checkpoint combinations:
# print(open_clip.list_pretrained())

# =============================================================================
# STEP 1 – Load PDF with Unstructured (high-resolution mode)
# =============================================================================
# UnstructuredLoader with strategy="hi_res" + extract_images_in_pdf=True will:
#   • Parse text, tables, headers etc. as individual elements.
#   • Detect embedded images, save them to disk, and store their file paths
#     in element.metadata["image_path"].
# mode="elements" keeps every detected block as a separate LangChain Document.

base_dir = os.path.dirname(__file__)                         # folder of this script
PDF_PATH = os.path.join(base_dir, "data/crag_paper.pdf")   # path to the target PDF

loader = UnstructuredLoader(
    PDF_PATH,
    mode="elements",            # each detected block → its own Document
    strategy="hi_res",          # high-quality OCR & layout analysis
    extract_images_in_pdf=True, # save images to disk and record their paths
)
elements = loader.load()   # list[Document] – one per detected element

# Quick summary of what was found
print(f"Loaded {len(elements)} elements|Documents\n")
# output: Loaded 254 elements|Documents

for cat in sorted(set(el.metadata.get("category", "unknown") for el in elements)):
    count = sum(1 for el in elements if el.metadata.get("category") == cat)
    print(" \nElements Found ")
    print(f"  {cat}: {count}")

# =============================================================================
# STEP 2 – Split text elements into overlapping chunks
# =============================================================================
# Image elements are excluded here because they don't have useful page_content
# (their content is their file path).  We only chunk text/table elements.

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # max characters per chunk
    chunk_overlap=200,  # characters shared between consecutive chunks (context continuity)
)

#filter out images and keep text_elements only from 254 element
text_elements = [el for el in elements if el.metadata.get("category") != "Image"]
text_docs = splitter.split_documents(text_elements)
print(f"Text chunks: {len(text_docs)}")

# =============================================================================
# STEP 3 – Create CLIP embeddings (shared text + image vector space)
# =============================================================================
# ViT-B-32 with the laion2b checkpoint can embed both a text string and an
# image into the same 512-dimensional vector space, allowing cross-modal
# similarity search (e.g. "find images that match this text query").

clip_embeddings = OpenCLIPEmbeddings(
    model_name="ViT-B-32",
    checkpoint="laion2b_s34b_b79k",  # pretrained weights from LAION-2B dataset
)

# =============================================================================
# STEP 4 – Build a Chroma vector store and index text chunks
# =============================================================================
# Chroma is chosen because it has first-class support for add_images(),
# which auto-encodes image files using the CLIP model.

vector_store = Chroma(
    embedding_function=clip_embeddings,
    collection_name="multimodal",   # logical namespace inside Chroma
)

# filter_complex_metadata removes any metadata values Chroma can't store
text_ids = vector_store.add_documents(filter_complex_metadata(text_docs))
print(f"Added {len(text_ids)} text chunks")

# =============================================================================
# STEP 5 – Index images directly (no captioning)
# =============================================================================
# add_images() accepts file URIs.  Chroma passes each image through the CLIP
# vision encoder and stores the resulting embedding.
# The image URI is saved as the Document's page_content for later retrieval.

image_elements = [
    el for el in elements
    if el.metadata.get("category") == "Image" and el.metadata.get("image_path")
]

image_uris      = [el.metadata["image_path"] for el in image_elements]
image_metadatas = [doc.metadata for doc in filter_complex_metadata(image_elements)]

image_ids = vector_store.add_images(uris=image_uris, metadatas=image_metadatas)
print(f"\nAdded {len(image_ids)} images")

# =============================================================================
# STEP 6 – Create a retriever
# =============================================================================
# The retriever will return up to k=6 documents (text chunks or image docs)
# whose embeddings are closest to the query embedding (also CLIP-encoded).

retriever = vector_store.as_retriever(search_kwargs={"k": 6})

# =============================================================================
# STEP 7 – Build the multimodal RAG chain
# =============================================================================

def encode_image(image_path: str) -> str:
    """Read an image file from disk and return it as a base64-encoded string.
    
    This is needed because OpenAI's vision API expects images as data-URLs
    (data:image/jpeg;base64,<data>) rather than raw file paths.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_messages(inputs):
    """Construct a multimodal HumanMessage from retrieved docs + the question.
    
    Retrieved documents are split into:
      • text_chunks  – regular text/table elements  → concatenated as context string
      • image_chunks – image documents (page_content = file URI) → base64-encoded
    
    The resulting OpenAI message has one text part followed by zero-or-more
    image_url parts, which GPT-4o-mini processes together.
    """
    docs     = inputs["docs"]
    question = inputs["question"]

    # Separate text and image documents from retriever results
    text_chunks  = [d for d in docs if d.metadata.get("category") != "Image"]
    image_chunks = [d for d in docs if d.metadata.get("category") == "Image"]

    # Build the plain-text context string from text chunks
    text_context = "\n\n".join(d.page_content for d in text_chunks)

    # Start building the message content list with the text+question part
    content = [
        {
            "type": "text",
            "text": (
                f"Answer the question based only on the following context:\n\n"
                f"{text_context}\n\nQuestion: {question}"
            ),
        },
    ]

    # Append each unique image as a base64 data-URL part
    # (page_content stores the image file URI when add_images() was used)
    seen = set()   # avoid adding the same image twice
    for doc in image_chunks:
        path = doc.page_content   # image URI set by add_images()
        if path and path not in seen:
            seen.add(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(path)}"
                },
            })

    return [HumanMessage(content=content)]   # single multimodal message


# Vision-capable LLM (gpt-4o-mini understands both text and image_url parts)
llm = ChatOpenAI(model="gpt-4o-mini")

# Full RAG chain:
#   1. Retrieve relevant docs (text + images) for the question using CLIP similarity.
#   2. Build a multimodal HumanMessage combining text context and base64 images.
#   3. Send to the LLM and parse its text response.
chain = (
    {"docs": retriever, "question": RunnablePassthrough()}  # step 1 – retrieve
    | RunnableLambda(build_messages)                        # step 2 – format
    | llm                                                   # step 3 – generate
    | StrOutputParser()                                     # extract text answer
)

# =============================================================================
# STEP 8 – Run example queries
# =============================================================================

# Query 1 – asks about a trend visible in a figure (tests image retrieval)
question = "How does generation accuracy change as retrieval accuracy drops for Self-CRAG and Self-RAG. explain in detail?"
answer = chain.invoke(question)
print(answer)


# Query 2 – asks for specific numeric data (tests table/text retrieval)
question = "Computational requirements of CRAG vs Self-RAG and which was has faster execution time and can you give me the actual TFLOPS values?"
answer = chain.invoke(question)
print(answer)