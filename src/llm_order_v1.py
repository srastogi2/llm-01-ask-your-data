
"""One function that sends the question to the LLM and returns its JSON query.

Works with Ollama or OpenAI: the .env file decides which.
"""

import json
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),  # Ollama ignores the key
)
MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")
print(f"Using model {MODEL}")


def get_completion(prompt, model=MODEL):
    messages = [{"role": "user", "content": prompt}]
    response =  client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def get_completion_from_messages(messages, model=MODEL, temperature=0):
    response =  client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
#     print(str(response.choices[0].message))
    return response.choices[0].message.content