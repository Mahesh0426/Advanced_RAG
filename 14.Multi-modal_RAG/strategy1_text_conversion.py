"""
Multimodal RAG - Strategy 1: Text Conversion (Image Captioning)
================================================================
This strategy converts images extracted from a PDF into text captions using a
Vision-Language Model (VLM). Both the text content and the generated captions
are embedded into a shared vector store for unified semantic retrieval.

Workflow:
    1. Load PDF with high-resolution extraction (text + images)
    2. Caption each extracted image using a VLM (GPT-4 Vision)
    3. Split text and caption documents into chunks
    4. Embed all chunks and store in a Chroma vector store
    5. At query time, retrieve relevant chunks and build a multimodal prompt
       that includes both text context and the raw images for the LLM
    6. Generate an answer grounded in both text and visual context
"""

import os
import base64
from dotenv import load_dotenv
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (e.g., OPENAI_API_KEY) from the .env file
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: PDF LOADING
# Uses UnstructuredLoader in "hi_res" mode to extract all content elements
# from the PDF including text, tables, and embedded images. Each element is
# tagged with a "category" (e.g., "NarrativeText", "Table", "Image").
# ─────────────────────────────────────────────────────────────────────────────

# Resolve the path to the target PDF relative to this script's location
base_dir = os.path.dirname(__file__)
PDF_PATH = os.path.join(base_dir, "data/crag_paper.pdf")

# 1. PDF Loading
# Load the PDF using high-resolution extraction strategy.
# - mode="elements"          → Instead of reading the PDF as one big block,it separates the PDF into different elements.
# - strategy="hi_res"        → uses a layout-aware model for accurate extraction
# - extract_images_in_pdf    → saves images to disk and records their file paths
loader = UnstructuredLoader(
    PDF_PATH,
    mode="elements",
    strategy="hi_res",       # Use high-resolution parsing.
    extract_images_in_pdf=True, # Save every image found inside the PDF.
)
elements = loader.load()

# Print a summary of all extracted element categories and their counts
print(f"Loaded {len(elements)} elements")
for cat in sorted(set(el.metadata.get("category", "unknown") for el in elements)):
    count = sum(1 for el in elements if el.metadata.get("category") == cat)
    print(f"  {cat}: {count}")

# Inspect metadata keys of the first element for reference
elements[0].metadata.keys()


# Debug: print the metadata of the first Image element to understand
# what fields are available (e.g., image_path, page_number, coordinates)
for element in elements:
    if element.metadata.get("category") == "Image":
        print(element.metadata.keys())
        print(element.metadata)
        break
    
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: IMAGE CAPTIONING WITH VISION-LANGUAGE MODEL (VLM)
# Each extracted image is encoded in base64 and sent to GPT-4 Vision.
# The model returns a rich text description capturing all visual details.
# These captions will later be embedded and used for semantic search.
# ─────────────────────────────────────────────────────────────────────────────

# Instantiate the Vision-Language Model used for image captioning
vlm = ChatOpenAI(model="gpt-5-mini")

# System prompt instructing the VLM how to describe images.
# The descriptions are optimized for semantic retrieval — they must be
# detailed enough for a user's natural-language query to match them.
IMAGE_CAPTION_SYSTEM_PROMPT = """You are a document analysis assistant. Your task is to generate \
detailed, accurate descriptions of images extracted from a document. These descriptions will be \
embedded into a vector store and used for semantic retrieval, so they must capture all information \
a user might search for.

For each image, describe:
- The image type (chart, diagram, photograph, table, illustration, screenshot, etc.)
- All visible text, labels, titles, captions, and annotations
- Key data, values, trends, or patterns (especially for charts and graphs)
- The main subject and all important visual elements
- Spatial relationships and structure where relevant

Be specific and thorough. Avoid vague language."""

def encode_image(image_path: str) -> str:
    """
    Read an image file from disk and encode it as a base64 string.

    Args:
        image_path: Absolute path to the image file on disk.

    Returns:
        A UTF-8 decoded base64 string representing the image bytes,
        suitable for embedding in a data URI (e.g., for API calls).
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def caption_image(image_path: str) -> str:
    """
    Generate a detailed text caption for an image using a VLM.

    Encodes the image as base64 and sends it to GPT-4 Vision along with a
    system prompt requesting a thorough description for semantic retrieval.

    Args:
        image_path: Absolute path to the image file on disk.

    Returns:
        A string containing the VLM-generated caption/description.
    """
    b64 = encode_image(image_path)
    messages = [
        SystemMessage(content=IMAGE_CAPTION_SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": "Describe this image extracted from a document."},
            # Pass the image inline as a base64-encoded data URI
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]),
    ]
    return vlm.invoke(messages).content

# Iterate over all extracted elements and caption every Image element.
# Each caption is stored as a new Document, preserving the original metadata
# (which includes the image_path needed later at inference time).
image_docs = []

for el in elements:
    if el.metadata.get("category") == "Image":
        image_path = el.metadata.get("image_path", "")
        if image_path:
            caption = caption_image(image_path)
            # Create a Document whose content is the VLM caption;
            # keep original metadata so image_path remains accessible
            image_docs.append(Document(page_content=caption, metadata=el.metadata))

print(f"Captioned {len(image_docs)} images")

# Preview the last generated caption for a sanity check
print(image_docs[-1].page_content)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: TEXT SPLITTING
# Both text elements and image captions are chunked into smaller pieces to
# stay within the context window and to improve retrieval precision.
# ─────────────────────────────────────────────────────────────────────────────

# Configure the text splitter with overlap to preserve context across chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Separate non-image elements (NarrativeText, Title, Table, etc.) from images
text_elements = [el for el in elements if el.metadata.get("category") != "Image"]

# Split text elements into overlapping chunks
text_docs = splitter.split_documents(text_elements)
# Split image captions into chunks (captions can be long for complex figures)
caption_docs = splitter.split_documents(image_docs)

# Combine all chunks into a single list for embedding
all_docs = text_docs + caption_docs
print(f"Total chunks: {len(all_docs)} ({len(text_docs)} text + {len(caption_docs)} captions)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: EMBEDDING & VECTOR STORE
# All document chunks (text + image captions) are embedded and stored in
# Chroma. filter_complex_metadata strips any non-serializable metadata fields
# that Chroma cannot handle (e.g., nested dicts, coordinate objects).
# ─────────────────────────────────────────────────────────────────────────────

# Initialize the OpenAI embedding model for converting text to vectors
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create a Chroma vector store from all document chunks.
# filter_complex_metadata ensures metadata is JSON-serializable for Chroma.
vector_store = Chroma.from_documents(filter_complex_metadata(all_docs), embeddings)

# Create a retriever that returns the top-4 most similar chunks for a query
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MULTIMODAL RAG CHAIN
# The chain retrieves relevant chunks, then builds a multimodal message that
# sends both the text context and any retrieved images directly to the LLM.
# This is the key insight of Strategy 1: captions enable text-based retrieval,
# but the original image is still used at inference time for accurate answers.
# ─────────────────────────────────────────────────────────────────────────────

# Base prompt template for text-only context (images are appended separately)
prompt = ChatPromptTemplate.from_messages([
    ("human", "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"),
])

def build_messages(inputs):
    """
    Construct the final multimodal message list for the LLM.

    Splits the retrieved documents into text chunks and image chunks.
    - Text chunks are formatted as a text context block in the prompt.
    - Image chunks have their original image files encoded as base64 and
      appended as inline image_url entries to the last HumanMessage.

    This allows the LLM to reason over both retrieved text AND visuals
    in a single call, combining the strengths of text search and vision.

    Args:
        inputs: dict with keys:
            - "docs"     : list of retrieved Document objects
            - "question" : the user's query string

    Returns:
        A list of LangChain message objects ready to pass to the LLM.
    """
    docs = inputs["docs"]
    question = inputs["question"]

    # Separate retrieved documents by type
    text_chunks = [d for d in docs if d.metadata.get("category") != "Image"]
    image_chunks = [d for d in docs if d.metadata.get("category") == "Image"]

    # Build a single text context string from all non-image chunks
    text_context = "\n\n".join(d.page_content for d in text_chunks)

    # Format the base prompt with the text context and question
    messages = prompt.format_messages(context=text_context, question=question)

    if image_chunks:
        seen_paths = set()   # Track paths to avoid sending the same image twice
        image_content = []
        for doc in image_chunks:
            image_path = doc.metadata.get("image_path")
            if image_path and image_path not in seen_paths:
                seen_paths.add(image_path)
                # Encode the image and format it as a vision API content block
                image_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"},
                })

        if image_content:
            # Replace the last HumanMessage with a multimodal version that
            # includes both the text prompt and the inline image(s)
            text_content = messages[-1].content
            messages[-1] = HumanMessage(content=[
                {"type": "text", "text": text_content},
                *image_content,   # Unpack all image content blocks
            ])

    return messages

# Instantiate the answer-generation LLM (same model, vision-capable)
llm = ChatOpenAI(model="gpt-5-mini")

# Build the full RAG chain using LangChain's LCEL (LangChain Expression Language):
# 1. Retrieve top-4 docs for the question via the vector store retriever
# 2. Build a multimodal message list with build_messages()
# 3. Pass the messages to the LLM and parse the string response
# 4. Also return the original context messages for inspection
chain = (
    {"docs": retriever, "question": RunnablePassthrough()}
    | RunnableLambda(build_messages)
    | {"response": llm | StrOutputParser(), "context": RunnablePassthrough()}
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: TESTING THE CHAIN
# Run two sample queries to verify the pipeline works end-to-end.
# The first query tests chart/figure understanding (visual reasoning).
# The second tests numerical fact retrieval that may require reading figures.
# ─────────────────────────────────────────────────────────────────────────────

# Query 1: Tests visual reasoning over a retrieved line chart
question = "How does Self-CRAG compares with Self-RAG as shown in the line chart. Can you explain this in a little bit more detail?"
answer = chain.invoke(question)
print(answer["response"])

# Inspect the length of the first context message content (for debugging)
len(answer["context"][0].content)

# Query 2: Tests numerical fact retrieval that may require reading figures/tables
question = "Computational requirements of CRAG vs Self-RAG and which was has faster execution time and can you give me the actual TFLOPS values?"
answer = chain.invoke(question)
print(answer["response"])