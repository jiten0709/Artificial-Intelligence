import os
from dotenv import load_dotenv
load_dotenv()

import openai
from pydantic import BaseModel, Field
import json
import requests

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------------
# Step 1: Define the tool (function) that we want to call
# ------------------------------------------

def get_weather(latitude, longitude):
    """This is a publicly available API that returns the weather for a given location."""
    
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()

    return data["current"]

# ------------------------------------------
# Step 2: Call model with get_weather tool defined
# ------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
        },
    }
]

system_prompt =  "You are a helpful weather assistant."

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": "What is the current temperature in Mumbai City?",
    },
]

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

# ------------------------------------------
# Step 3: model decides to call function(s)
# ------------------------------------------

# (Optional) Print the raw completion for debugging
# print(completion.model_dump())

# ------------------------------------------
# Step 4: Execute get_weather function
# ------------------------------------------

def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    
for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)
    result = call_function(name, args)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        }
    )

# ------------------------------------------
# Step 5: Supply result and call model again with JSON response format
# ------------------------------------------

class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )

completion_2 = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format={"type": "json_object"},
)

# ------------------------------------------
# Step 6: Check model response
# ------------------------------------------

response_json = completion_2.choices[0].message.content
final_response = WeatherResponse(**json.loads(response_json))