from guardrails import Guard
from guardrails_grhub_toxic_language import ToxicLanguage
from dotenv import load_dotenv

# load the env vars
load_dotenv()

# create the validator
validator = ToxicLanguage(on_fail="reask")

# create guard and add the validator to the guard
guard = Guard().use(validator)

# validate the user input

user_input_1 = "Great job! You are doing amazing work. There were times when you got stuck but you still managed to get the work done. Kudos to you!"
user_input_2 = "Great job! You are doing amazing work. There were times when you got stuck but you still managed to get the work done. If you were not this stupid, the work would have been done a lot quicker!"

response_1 = guard.validate(user_input_1)

print("response_1 ➡️:",response_1)

print(f"Validation Result: {response_1.validation_passed}")
print(f"Raw User Input: {response_1.raw_llm_output}")
print(f"Validated Output: {response_1.validated_output}")

# validate the user input

response_2 = guard.validate(user_input_2)

print("response_2 ➡️:",response_2)

print(f"Validation Result: {response_2.validation_passed}")
print(f"Raw User Input: {response_2.raw_llm_output}")
print(f"Validated Output: {response_2.validated_output}")

print(response_2.reask)