
from guardrails import Guard
from guardrails_grhub_competitor_check import CompetitorCheck

from dotenv import load_dotenv

# load the env vars
load_dotenv()

# create the validator
competitor_list = ["Apple", "Google", "Microsoft", "Samsung", "Amazon"]

validator = CompetitorCheck(competitors=competitor_list,
                            on_fail="exception")

# create guard and add the validator to the guard
guard = Guard().use(validator)

# validate the user input
user_input_1 = "I like all the fruits, but apple and grapes are my favorite."
user_input_2 = "Apple is great, makes durable and high quality products."

response_1 = guard.validate(user_input_1)
print("response_1:",response_1)

print(f"\nValidation Result: {response_1.validation_passed}")
print(f"Raw User Input: {response_1.raw_llm_output}")
print(f"Validated Output: {response_1.validated_output}")

try: 
    response_2 = guard.validate(user_input_2)
except Exception as e:
    print(e)
