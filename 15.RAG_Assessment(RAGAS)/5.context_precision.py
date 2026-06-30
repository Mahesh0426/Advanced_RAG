# Context Precision : measures whether the context contains information necessary for answering the question
# Low score  = Bad  — context is noisy or irrelevant
# High score = Good — context is focused and directly supports answering the question

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import ContextPrecision
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
llm = llm_factory("gpt-5-mini", client=client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

scorer = ContextPrecision(llm=llm, embeddings=embeddings)

# Example 1 : Relevant context is first but two irrelevant chunks follow
async def main1():
    result = await scorer.ascore(
    user_input="What is photosynthesis?",
    reference="Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
    retrieved_contexts=[
        "Photosynthesis is the biological process through which plants convert sunlight and carbon dioxide into glucose and oxygen.",
        "Plants require water and minerals from the soil for growth.",
        "The Amazon rainforest is home to thousands of plant species."
    ]
)
    print(f"Context Precision Score: {result.value}")
    # Context Precision Score: 0.9999999999
  
      

asyncio.run(main1())

# Example 2 : All retrieved contexts are relevant and support the reference answer
async def main2():
    result = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris, France.",
    retrieved_contexts=[
        "The Eiffel Tower is located on the Champ de Mars in Paris, France.",
        "The Eiffel Tower was built in 1889 and stands 330 meters tall in Paris."
    ]
)
    print(f"Context Precision Score 2: {result.value}")
    # Context Precision Score 2: 0.99999999995

asyncio.run(main2())


# Example 3 : Two completely irrelevant chunks are ranked before the one relevant chunk
async def main3():
    result = await scorer.ascore(
    user_input="What is the boiling point of water?",
    reference="Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
    retrieved_contexts=[
        "The capital of France is Paris, a major European city known for the Eiffel Tower.",
        "The density of water is 1 gram per cubic centimeter at 4 degrees Celsius.",
        "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."
    ]
)
    print(f"Context Precision Score 3: {result.value}")
    # Context Precision Score: 0.3333333333

asyncio.run(main3())