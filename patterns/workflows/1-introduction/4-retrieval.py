import os
from dotenv import load_dotenv
load_dotenv()

import openai
import json
from pydantic import BaseModel, Field

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------------------
# Define the knowledge base retrieval tool
# --------------------------------------------------------------

def search_kb(question: str):
    """
    Load the whole knowledge base from the JSON file.
    (This is a mock function for demonstration purposes, we don't search)
    """
    
    with open("kb.json", "r") as f:
        return json.load(f)
    
# --------------------------------------------------------------
# Step 1: Call model with search_kb tool defined
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        }
    }
]

system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"},
]

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

# --------------------------------------------------------------
# Step 2: Model decides to call function(s)
# --------------------------------------------------------------

# if want to print the raw completion
# print(completion.model_dump())

# --------------------------------------------------------------
# Step 3: Execute search_kb function
# --------------------------------------------------------------

def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)
    else:
        raise ValueError(f"Unknown function: {name}")
    
for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)
    result = call_function(name, args)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        }
    )

# --------------------------------------------------------------
# Step 4: supply result and call model again with json response
# ---------------------------------------------------------------

class KBResponse(BaseModel):
    answer: str = Field(
        description="The answer to the user's question."
    )
    source: int = Field(
        description="The record id of the answer."
    )

completion_2 = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format={"type": "json_object"},
)

# --------------------------------------------------------------
# Step 5: Check model response
# --------------------------------------------------------------

response_json = completion_2.choices[0].message.content
final_resonse = KBResponse(**json.loads(response_json))
print("FINAL RESPONSE ANSWER (valid user content): ", final_resonse.answer)
print("FINAL RESPONSE SOURCE (valid user content): ", final_resonse.source)

# --------------------------------------------------------------
# Question that doesn't trigger the tool
# --------------------------------------------------------------

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the store's address?"},
]

completion_3 = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format={"type": "json_object"},
)

print("FINAL RESPONSE ANSWER (invalid user content): ", completion_3.choices[0].message.content)