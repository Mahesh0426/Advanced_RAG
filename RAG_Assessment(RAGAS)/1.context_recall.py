# Context Recall : measures whether the retrieved context necessary for answering the question
# Low score = covers few key information
# High score = covers all key information


from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall
import os
import asyncio


load_dotenv()

# Initialize OpenAI client for RAGAS
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
llm = llm_factory("gpt-4o-mini", client=client, max_tokens=1024)
scorer = ContextRecall(llm=llm)


async def main():
    # Example : 1 context cover system but miss the cause (Insulin resistance / obesity)
    result = await scorer.ascore(
        user_input="What are the symptoms and cause of type 2 diabetes?",
        retrieved_contexts=[
            "Type 2 diabetes symptoms include frequent, urination and excessive thirst"
            "People with type 2 diabetes often experience fatigue and blood vision."
        ],
        reference="Type 2 diabetes is caused by insulin resistance, often linked to obesity and a sedentary lifestyle. Its symptoms include frequent urination, excessive thirst, fatigue, blurred vision, and slow-healing sores."
    )
    print(f"Context Recall Score 1: {result.value}")
    # Context Recall Score: 0.5714285714285714

asyncio.run(main())


async def main2():
    # Example 2 : The retrieved context fully covers every claim in the ground truth
    result = await scorer.ascore(
        user_input="Where is the Eiffel Tower located?",
        retrieved_contexts=[
            "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France."
        ],
        reference="The Eiffel Tower is located in Paris, France."
    )
    print(f"Context Recall Score 2: {result.value}")
    # Context Recall Score: 1.0

asyncio.run(main2())

# Example 3: Context only addresses treatment; ground truth covers causes and symptoms
async def main3():
    result = await scorer.ascore(
    user_input="What causes and characterizes Parkinson's disease?",
    retrieved_contexts=[
        "Parkinson's disease is managed using medications such as levodopa and physical therapy to improve quality of life."
    ],
    reference="Parkinson's disease is caused by the loss of dopamine-producing neurons in the brain. It is characterized by tremors, stiffness, slowness of movement, and balance problems."
    )
    print(f"Context Recall Score 3: {result.value}")
    # Context Recall Score: 0.0

asyncio.run(main3())


