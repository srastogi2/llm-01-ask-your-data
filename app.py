"""Ask questions about a CSV in plain English.

Run:  python app.py
Quit: type 'exit'
"""

import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # must run before src.llm reads the settings

from src.llm import question_to_query  # noqa: E402
from src.prompts import build_system_prompt  # noqa: E402
from src.query import QueryError, describe, run, validate  # noqa: E402
from src.schema import describe_schema, queryable_columns  # noqa: E402


def format_number(value) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def format_result(result) -> str:
    if isinstance(result, pd.Series):
        return "\n" + "\n".join(f"  {idx}: {format_number(val)}" for idx, val in result.items())
    return format_number(result)


def main():
    data_path = os.getenv("DATA_PATH", "data/lending_club.csv")
    df = pd.read_csv(data_path)
    schema_text = describe_schema(df)
    allowed = queryable_columns(df)
    system_prompt = build_system_prompt(schema_text)

    print(f"Loaded {len(df):,} rows from {data_path}. Ask a question, or type 'exit'.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            query, stats = question_to_query(system_prompt, question)
            query = validate(query, df, allowed)
            if not query.get("answerable", True):
                print(f"Bot: I can't answer that from this data. {query.get('reason', '')}")
            else:
                result = run(query, df)
                print(f"Bot: {format_result(result)}")
                print(f"     Query used: {describe(query)}")
            print(f"     [{stats['model']} | {stats['seconds']}s | "
                  f"{stats['input_tokens']} in / {stats['output_tokens']} out tokens]\n")
        except (QueryError, ValueError) as e:
            print(f"Bot: Sorry, I couldn't turn that into a valid query. ({e})\n")


if __name__ == "__main__":
    main()
