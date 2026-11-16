"""The system prompt: instructions, the query format, and few-shot examples.

Edit the examples to match your dataset's real columns. Good examples are
the single biggest lever on accuracy, especially for small local models.
"""

SYSTEM_PROMPT = """You translate questions about a dataset into a JSON query.
You never calculate answers yourself. Reply with JSON only, no other text.

Dataset columns:
{schema}

JSON format:
{{
  "answerable": true or false,
  "operation": "count" | "mean" | "sum" | "min" | "max",
  "column": column to aggregate, or null for count,
  "filters": [{{"column": name, "op": "==" | "!=" | ">" | ">=" | "<" | "<=", "value": value}}],
  "group_by": column name or null,
  "top_n": number of groups to return (largest first), or null,
  "reason": short explanation if answerable is false, else ""
}}

Rules:
- Only use columns listed above. Category values must match exactly.
- If the question cannot be answered with these columns and operations,
  set "answerable" to false and explain why in "reason".

Examples:

Question: How many loans are there?
{{"answerable": true, "operation": "count", "column": null, "filters": [], "group_by": null, "top_n": null, "reason": ""}}

Question: What is the average loan amount for borrowers who rent?
{{"answerable": true, "operation": "mean", "column": "loan_amnt", "filters": [{{"column": "home_ownership", "op": "==", "value": "RENT"}}], "group_by": null, "top_n": null, "reason": ""}}

Question: Which 3 loan purposes have the highest total loan amount?
{{"answerable": true, "operation": "sum", "column": "loan_amnt", "filters": [], "group_by": "purpose", "top_n": 3, "reason": ""}}

Question: Will this customer repay their loan next year?
{{"answerable": false, "operation": "count", "column": null, "filters": [], "group_by": null, "top_n": null, "reason": "Predicting individual outcomes needs a model, not a data query."}}
"""


def build_system_prompt(schema_text: str) -> str:
    return SYSTEM_PROMPT.format(schema=schema_text)
