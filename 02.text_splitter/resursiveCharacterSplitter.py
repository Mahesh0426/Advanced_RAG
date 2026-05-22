
#CharacterTextSplitter  - You know exactly where your text should be split and want full control.
# vs
#  RecursiveCharacterTextSplitter - You're dealing with normal human-written text — articles, PDFs, docs, web pages.


from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

text = """AI is transforming technology.
Machine learning is growing fast.

Deep learning models process data.
NLP has made huge strides recently."""

print("=" * 50)
print("CharacterTextSplitter (separator='\\n\\n' only)")
print("=" * 50)
char_splitter = CharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=0,
    separator="\n\n"
)
for i, chunk in enumerate(char_splitter.split_text(text), 1):
    print(f"Chunk {i} ({len(chunk)} chars): {repr(chunk)}")

print()
print("=" * 50)
print("RecursiveCharacterTextSplitter")
print("=" * 50)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=0
)
for i, chunk in enumerate(recursive_splitter.split_text(text), 1):
    print(f"Chunk {i} ({len(chunk)} chars): {repr(chunk)}")