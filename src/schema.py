"""Turn a DataFrame into a short text description the LLM can read.

The LLM never sees the data itself, only this summary: column names,
types, and either the allowed values (for categories) or the range
(for numbers). Keeping it short matters for small local models.
"""

import pandas as pd

MAX_CATEGORIES = 15  # list values only for low-cardinality text columns


def describe_schema(df: pd.DataFrame) -> str:
    lines = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            lines.append(
                f"- {col} (number, range {series.min():g} to {series.max():g})"
            )
        else:
            values = series.dropna().unique()
            if len(values) <= MAX_CATEGORIES:
                shown = ", ".join(str(v) for v in sorted(values, key=str))
                lines.append(f"- {col} (category: {shown})")
            # High-cardinality text (names, addresses) is left out on purpose:
            # the bot can't filter on it reliably, so it shouldn't try.
    return "\n".join(lines)


def queryable_columns(df: pd.DataFrame) -> set[str]:
    """Columns that appear in the schema, so the only ones a query may use."""
    cols = set()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.add(col)
        elif df[col].dropna().nunique() <= MAX_CATEGORIES:
            cols.add(col)
    return cols
