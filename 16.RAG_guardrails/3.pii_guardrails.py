from guardrails import Guard
from guardrails_grhub_guardrails_pii import GuardrailsPII
from dotenv import load_dotenv
# load the env vars
load_dotenv()

# create the validator
validator = GuardrailsPII(entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="fix")  # attempt to fix the user input if the validator fails

# create guard and add the validator to the guard
guard = Guard().use(validator)

# validate the user input

user_input_1 = "Write an official mail to the RTO office regarding some issues in my driving license"
user_input_2 = "Write an official mail to the RTO office regarding some issues in my driving license. My email is amitkr123@gmail.com and my phone number is 8920333365."

response_1 = guard.validate(user_input_1)

print(f"Validation Result: {response_1.validation_passed}")
print(f"Raw User Input: {response_1.raw_llm_output}")
print(f"Validated Output: {response_1.validated_output}")

# validate the user input

response_2 = guard.validate(user_input_2)

response_2


print(f"Validation Result: {response_2.validation_passed}")
print(f"Raw User Input: {response_2.raw_llm_output}")
print(f"Validated Output: {response_2.validated_output}")