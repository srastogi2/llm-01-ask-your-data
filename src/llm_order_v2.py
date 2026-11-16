"""LLM client used by OrderBot.

The defaults connect to a local Ollama server through its OpenAI-compatible API.
Environment variables can override the endpoint, API key, and model.
"""

import os

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # .env loading is optional; environment variables can also be set by the shell.
    pass


BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=120.0,
)

print(f"Using model: {MODEL}")
print(f"LLM endpoint: {BASE_URL}")


def get_completion(prompt, model=MODEL, temperature=0):
    """Send a single user prompt and return the assistant's text."""
    messages = [{"role": "user", "content": prompt}]
    return get_completion_from_messages(
        messages,
        model=model,
        temperature=temperature,
    )


def get_completion_from_messages(messages, model=MODEL, temperature=0):
    """Send a conversation and return the assistant's text."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("The language model returned an empty response.")

    return content
