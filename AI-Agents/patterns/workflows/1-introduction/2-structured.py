import os
from dotenv import load_dotenv
load_dotenv()

import openai
from pydantic import BaseModel
import json

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------------------
# step 1: Define the response format in a Pydantic model
# ----------------------------------------

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

# ----------------------------------------
# step 2: Call the model with JSON response format
# ----------------------------------------

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract the event information and respond as a JSON object with keys: name, date, participants."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."}
    ],
    response_format={"type": "json_object"},
)

# ----------------------------------------
# step 3: Parse the response
# ----------------------------------------

response_json = completion.choices[0].message.content
event = CalendarEvent(**json.loads(response_json))

print(event.name)          # Alice and Bob are going to a science fair
print(event.date)         # Friday
print(event.participants)  # ['Alice', 'Bob']
