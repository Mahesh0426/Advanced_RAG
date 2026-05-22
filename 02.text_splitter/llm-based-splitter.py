from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

text = """Artificial intelligence is transforming technology and shaping the future.
Machine learning algorithms are becoming more sophisticated every day.
Deep learning models can now process vast amounts of data efficiently.
Neural networks are inspired by the human brain's structure.
The best pasta recipes include fresh ingredients and proper cooking techniques.
Italian cuisine emphasizes quality olive oil and regional cheeses.
Authentic carbonara uses guanciale, eggs, pecorino romano, and black pepper.
Cooking pasta al dente ensures the best texture and flavor.
Climate change is affecting ecosystems worldwide.
Rising temperatures are causing glaciers to melt at unprecedented rates.
Scientists warn that immediate action is needed to reduce carbon emissions.
Renewable energy sources offer hope for a sustainable future."""



# pydantic class for structured output
class Chunk(BaseModel): 
    
    chunk_text: str
    summary: str
    
    
class Chunker(BaseModel):
    
    chunks: list[Chunk]


# define model
model = ChatOpenAI(model="gpt-5-mini")
llm_chunker = model.with_structured_output(schema=Chunker)

# prompt for chunking
prompt = ChatPromptTemplate(messages=[
    ("system", 
     """You are an expert Text Chunker that splits the given text and outputs them as a 
     list of strings. You understand the natural topic boundaries of text and 
     also do not change the existing text. You just split the text where ever applicable.
     Once you create the chunk, you also generate a 1-2 line summary of the chunk also"""),
    ("human",
     "Split the given text into chunks\nText: {text}")
], input_variables=["text"])

# chunking through llm
model_chain = prompt | llm_chunker
response = model_chain.invoke({"text": text})
# print(response)
print(response.chunks)
print(len(response.chunks))

chunks = response.chunks
from termcolor import COLORS, colored
from random import choice


def display_chunks(chunks):
    colors_list = list(COLORS.keys())[2:8]
    print(f"Total Number of Chunks: {len(chunks)}")
    
    for num, chunk in enumerate(chunks, 1):
        print(f"Chunk {num}: Length {len(chunk)} chars")
        print(colored(text=chunk, color=choice(colors_list)), end="\n\n")


display_chunks([chunk.chunk_text for chunk in chunks])