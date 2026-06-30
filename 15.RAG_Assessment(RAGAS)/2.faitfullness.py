# Faithfulness: measures whether the response is factually consistent with the provided reference text
# Low score =  bad
# High score = good

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client for RAGAS
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
llm = llm_factory("gpt-4o-mini", client=client, max_tokens=1024)
scorer = Faithfulness(llm=llm)

# Example 1: Response is mostly grounded but adds one unsupported claim
# 'has been proven to prevent all forms of cancer' is not in the context
async def main1():
    result = await scorer.ascore(
    user_input="What are the health benefits of green tea?",
    response="Green tea contains antioxidants that help reduce inflammation. It also boosts metabolism and has been proven to prevent all forms of cancer.",
    retrieved_contexts=[
        "Green tea is rich in antioxidants, particularly catechins, which help reduce inflammation and oxidative stress.",
        "Studies suggest green tea may modestly boost metabolic rate."
    ]
)
    print(f"Faithfulness Score 1 : {result.value}")
    # Faithfulness Score 1: 0.5

asyncio.run(main1())

# Example 2 : Every claim in the response is directly supported by the context
async def main2():
    result = await scorer.ascore(
    user_input="When was the first Super Bowl played?",
    response="The first Super Bowl was played on January 15, 1967, at the Los Angeles Memorial Coliseum.",
    retrieved_contexts=[
        "The First AFL-NFL World Championship Game was played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles, California."
    ]
)
    print(f"Faithfulness Score 2 : {result.value}")
    # Faithfulness Score 2 : 1.0

asyncio.run(main2())



# Example 3: Response introduces multiple facts not found in the context
# Context only states the speed; travel time to Earth and Earth-circling claim are hallucinated
async def main3():
    result = await scorer.ascore(
    user_input="What is the speed of light?",
    response="The speed of light is approximately 3x10^8 meters per second. It takes light about 8 minutes to travel from the Sun to Earth, and light can circle the Earth 7.5 times in one second.",
    retrieved_contexts=[
        "The speed of light in a vacuum is approximately 299,792,458 meters per second."
    ]
)
    print(f"Faithfulness Score 3 : {result.value}")
    # Faithfulness Score 3 : 0.3333333333333333

asyncio.run(main3())