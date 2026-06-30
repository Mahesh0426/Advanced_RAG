from guardrails import Guard
from guardrails_grhub_response_evaluator import ResponseEvaluator
from dotenv import load_dotenv

# load the env vars
load_dotenv()

# create the validator
validator = ResponseEvaluator(llm_callable="gpt-5-mini",
                              on_fail="noop")

# create guard and add the validator to the guard
guard = Guard().use(validator)

# validate the user input

user_query = "What is the capital of India?"

llm_response_1 = "The capital of India is New Delhi."
llm_response_2 = "India is known for its rich history and culture. It has a city with large metro rail network and is one of the most populous cities in the world."

response_1 = guard.validate(llm_response_1, 
                            metadata={"validation_question": user_query})

print("response_1",response_1)

print(f"Validation Result: {response_1.validation_passed}")
print(f"Raw User Input: {response_1.raw_llm_output}")
print(f"Validated Output: {response_1.validated_output}")


response_2 = guard.validate(llm_response_2, 
                            metadata={"validation_question": user_query})


print("response_2",response_2)

print(f"Validation Result: {response_2.validation_passed}")
print(f"Raw User Input: {response_2.raw_llm_output}")
print(f"Validated Output: {response_2.validated_output}")