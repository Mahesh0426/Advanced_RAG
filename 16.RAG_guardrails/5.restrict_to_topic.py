from guardrails import Guard
from tryolabs_grhub_restricttotopic import RestrictToTopic
from dotenv import load_dotenv

# load the env vars
load_dotenv()

# create the validator
validator = RestrictToTopic(valid_topics=["machine learning", "artificial intelligence", "data science"],
                            invalid_topics=["sports", "politics", "entertainment"],
                            on_fail="exception")

# create guard and add the validator to the guard
guard = Guard().use(validator)

# validate the user input

user_input_1 = "Can you tell me what is Linear Regression?"
user_input_2 = "Who won the IPL final in 2026?"

response_1 = guard.validate(user_input_1)

print("response_1 ➡️:",response_1)

print(f"Validation Result: {response_1.validation_passed}")
print(f"Raw User Input: {response_1.raw_llm_output}")
print(f"Validated Output: {response_1.validated_output}")

# validate the user input

response_2 = guard.validate(user_input_2)

print("response_2 ➡️:",response_2)

try:
    response_2 = guard.validate(user_input_2)
except Exception as e:
    print(e)