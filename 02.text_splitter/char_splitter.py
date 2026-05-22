from langchain_text_splitters import CharacterTextSplitter

# STEP 1 — What is a CharacterTextSplitter?   
#  It cuts long text into smaller "chunks" of fixed size. 

text = """Artificial intelligence is transforming technology and shaping the future. 
Machine learning algorithms are becoming more sophisticated every day.
Deep learning models can now process vast amounts of data efficiently.

Natural language processing has made significant strides in recent years.
Computer vision systems can now identify objects with remarkable accuracy.
Reinforcement learning is enabling robots to learn complex tasks autonomously.

The impact of AI extends across multiple industries including healthcare, finance, and transportation.
Ethical considerations around AI development are becoming increasingly important.
Researchers are working on making AI systems more transparent and explainable."""



# create the splitter
splitter = CharacterTextSplitter(
    chunk_size=100,       # each chunk is AT MOST 100 characters
    chunk_overlap=0,      # no repeated text between chunks
    length_function=len,  # measure length by character count
    separator=""          # no preferred split point — split anywhere
)

# split the text \ most cases you will use split_documents()
chunks = splitter.split_text(text)
# print(chunks)

print(f"\nNumber of Chunks: {len(chunks)}")
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk)
 

#  STEP 2 — Colorful chunk display using termcolor   
print("\n" + "="*60)
print("STEP 2: Display chunks in random colors (termcolor)")
print("="*60)

from termcolor import COLORS, colored
from random import choice

#  COLORS is a dict like: {'reset': '\033[0m', 'bold': '\033[01m', 'red': ...}
# We skip the first 2 entries (reset/bold) and take 6 color names

colors_list = list(COLORS.keys())[2:8]
print(f"\nAvailable colors we'll use: {colors_list}")

def display_chunks(chunks):
    colors_list = list(COLORS.keys())[2:8]
    print(f"\nTotal Number of Chunks: {len(chunks)}")
    for num, chunk in enumerate(chunks, 1):
        print(f"\nChunk {num}: Length {len(chunk)} chars")
        print(colored(text=chunk, color=choice(colors_list)), end="\n\n")
 
# display_chunks(chunks)

# STEP 3 — Reusable create_chunks() helper      
 
print("\n" + "="*60)
print("STEP 3: create_chunks() helper — change settings easily")
print("="*60)
 
def create_chunks(text: str, chunk_size: int,
                  separator: str, chunk_overlap: int = 0) -> list[str]:
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator
    )
    return splitter.split_text(text=text)
 
# Same as Step 1 but using our helper
# display_chunks(create_chunks(text, 100, ""))

# STEP 4 — Effect of chunk_size: bigger = fewer chunks 

 
print("\n" + "="*60)
print("STEP 4: Larger chunk_size=300 → fewer, bigger chunks")
print("="*60)
 
# display_chunks(create_chunks(text, 300, ""))

 

# STEP 5 — separator=" "  → split on word boundaries     
 
print("\n" + "="*60)
print("STEP 5: separator=' ' → no more mid-word splits!")
print("="*60)
 
display_chunks(create_chunks(text, 100, " ", 20))
 