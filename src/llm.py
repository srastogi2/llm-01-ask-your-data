"""One function that sends the question to the LLM and returns its JSON query.

Works with Ollama or OpenAI: the .env file decides which.
"""

import json
import os
import time

from openai import OpenAI

# client = OpenAI(
#     base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
#     # api_key=os.getenv("LLM_API_KEY", "ollama"),  # Ollama ignores the key
#     api_key=os.getenv("LLM_API_KEY", 'OPENAI_API_KEY'),  # Ollama ignores the key
# )

client = OpenAI(
    base_url=os.getenv("GPT_LLM_BASE_URL", "http://localhost:11434/v1"),
    # api_key=os.getenv("LLM_API_KEY", "ollama"),  # Ollama ignores the key
    api_key=os.getenv("GPT_LLM_API_KEY", 'OPENAI_API_KEY'),  # Ollama ignores the key
)
api_key=os.getenv("GPT_LLM_API_KEY", 'OPENAI_API_KEY')
MODEL = os.getenv("GPT_LLM_MODEL", "qwen2.5-coder:7b")
print(f"Using model {MODEL} & {api_key}")

# import ollama

# client = ollama.Client(host="http://localhost:11434")
# MODEL = os.getenv("LLM_MODEL","qwen2.5-coder:7b")

# def question_to_query(system_prompt: str, question: str) -> tuple[dict, dict]:
#     """Returns (query, stats). Raises ValueError if the reply isn't valid JSON."""
#     start = time.perf_counter()
#     response = client.chat(
#         model=MODEL,
#         response_format={"type": "json_object"},
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f"Question: {question}"},
#         ],
#         options={"temperature:0"}
#     )
#     seconds = time.perf_counter() - start

#     text = response.message.content
#     try:
#         query = json.loads(text)
#     except json.JSONDecodeError:
#         raise ValueError(f"The model didn't return valid JSON: {text[:200]}")

#     usage = response.usage
#     stats = {
#         "model": MODEL,
#         "seconds": round(seconds, 2),
#         "input_tokens": usage.prompt_tokens if usage else None,
#         "output_tokens": usage.completion_tokens if usage else None,
#     }
#     return query, stats


def question_to_query(system_prompt: str, question: str) -> tuple[dict, dict]:
    """Returns (query, stats). Raises ValueError if the reply isn't valid JSON."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        temperature=1,  # same question, same answer
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ],
    )
    seconds = time.perf_counter() - start

    text = response.choices[0].message.content
    try:
        query = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"The model didn't return valid JSON: {text[:200]}")

    usage = response.usage
    stats = {
        "model": MODEL,
        "seconds": round(seconds, 2),
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
    }
    return query, stats
