# Noise Sensitivity : measures whether the response is influenced by irrelevant/noisy contexts in retrieved_contexts
# Low score  = Good — the response ignores noise and stays grounded in relevant context and reference
# High score = Bad  — the response is misled by irrelevant contexts and introduces hallucinated/off-topic claims

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import NoiseSensitivity
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
llm = llm_factory("gpt-5-mini", client=client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

scorer = NoiseSensitivity(llm=llm, embeddings=embeddings)

# Example 1 : Response answers the question but drifts into tangential information
# Mentioning the compromise between Sydney and Melbourne dilutes relevancy
async def main1():
    result = await scorer.ascore(
        user_input="What is the Life Insurance Corporation of India (LIC) known for?",
    response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.",
    reference="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.",
    retrieved_contexts=[
        "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
        "LIC is the largest insurance company in India, with a vast network of policyholders and huge investments.",
        "As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.",
        "The Indian economy is one of the fastest-growing major economies in the world, thanks to sectors like finance, technology, and manufacturing."
    ]
    )
    print(f"Noise Sensitivity Score 1 : {result.value}")
    # Noise Sensitivity Score: 0.3333333333333333

asyncio.run(main1())

# Example 2: Response is direct and precisely answers the question with no filler
async def main2():
    result = await scorer.ascore(
     user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris, France.",
    reference="The Eiffel Tower is located in Paris, France.",
    retrieved_contexts=[
        "The Eiffel Tower is a landmark located in Paris, France.",
        "Paris is also home to the Louvre Museum, which contains thousands of artworks including the Mona Lisa.",
        "France is known for its cuisine, wine, and fashion industry."
    ]
    )
    print(f"Noise Sensitivity Score 2 : {result.value}")
    # Noise Sensitivity Score: 0.0

asyncio.run(main2())


