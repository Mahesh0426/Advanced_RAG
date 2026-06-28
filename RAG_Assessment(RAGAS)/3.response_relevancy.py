# Response Relevancy : measures how well the response addresses the actual user question (no irrelevant or missing content)
# Low score  = response is off-topic, evasive, or fails to directly answer the user's question
# High score = response is concise, directly addresses the question, and contains no tangential/irrelevant information

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerRelevancy
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)

# Example 1 : Response answers the question but drifts into tangential information
# Mentioning the compromise between Sydney and Melbourne dilutes relevancy
async def main1():
    result = await scorer.ascore(
        user_input="What is the capital of Australia?",
        response="Australia is a large country in the Southern Hemisphere. It has many major cities including Sydney, Melbourne, and Brisbane. Canberra is the capital city, chosen as a compromise between Sydney and Melbourne. Australia also has a diverse economy driven by mining and agriculture."
    )
    print(f"Response Relevancy Score 1 : {result.value}")
    # Response Relevancy Score: 1.0000000000000002

asyncio.run(main1())

# Example 2: Response is direct and precisely answers the question with no filler
async def main2():
    result = await scorer.ascore(
    user_input="When was the first Super Bowl played?",
    response="The first Super Bowl was played on January 15, 1967."
    )
    print(f"Response Relevancy Score 2 : {result.value}")
    # Response Relevancy Score: 0.26754335611869756

asyncio.run(main2())


# Example 3 : Response talks about water as a topic but never states its boiling point
async def main3():
    result = await scorer.ascore(
        user_input="What is the boiling point of water?",
        response="Water is a fascinating substance found all over the Earth. It is essential for all known forms of life and covers about 71 percent of the Earth's surface. Water is found in oceans, rivers, lakes, and glaciers and plays a key role in regulating climate."
    )
    print(f"Response Relevancy Score 3 : {result.value}")

asyncio.run(main3())