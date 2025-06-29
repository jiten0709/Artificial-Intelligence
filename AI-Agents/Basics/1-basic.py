import os
from dotenv import load_dotenv
load_dotenv()

import openai

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}
    ]
)

print(response.choices[0].message.content)