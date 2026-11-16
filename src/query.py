"""Validate the LLM's JSON query, then run it with pandas.

This is the safety layer: the LLM's output is treated as untrusted input.
Anything it gets wrong is caught here with a clear message, and no
LLM-generated code is ever executed.
"""

import pandas as pd

OPERATIONS = {"count", "mean", "sum", "min", "max"}
COMPARISONS = {"==", "!=", ">", ">=", "<", "<="}


class QueryError(ValueError):
    """The LLM produced a query we can't safely run."""


def validate(query: dict, df: pd.DataFrame, allowed_columns: set[str]) -> dict:
    if not query.get("answerable", True):
        return query

    op = query.get("operation")
    if op not in OPERATIONS:
        raise QueryError(f"Unknown operation: {op!r}")

    column = query.get("column")
    if op != "count":
        if column not in allowed_columns:
            raise QueryError(f"Unknown column: {column!r}")
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise QueryError(f"Can't take the {op} of a text column ({column})")

    for f in query.get("filters") or []:
        col, cmp = f.get("column"), f.get("op")
        if col not in allowed_columns:
            raise QueryError(f"Unknown filter column: {col!r}")
        if cmp not in COMPARISONS:
            raise QueryError(f"Unknown comparison: {cmp!r}")
        f["value"] = _match_value(df[col], f.get("value"))

    group_by = query.get("group_by")
    if group_by is not None and group_by not in allowed_columns:
        raise QueryError(f"Unknown group-by column: {group_by!r}")

    return query


def _match_value(series: pd.Series, value):
    """Cast the value to the column's type; fix case for category values."""
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(value)
        except (TypeError, ValueError):
            raise QueryError(f"Expected a number for {series.name}, got {value!r}")
    lookup = {str(v).lower(): v for v in series.dropna().unique()}
    match = lookup.get(str(value).lower())
    if match is None:
        raise QueryError(f"{value!r} is not a value of {series.name}")
    return match


def run(query: dict, df: pd.DataFrame):
    data = df
    for f in query.get("filters") or []:
        col, cmp, val = f["column"], f["op"], f["value"]
        data = data[_compare(data[col], cmp, val)]

    op, column, group_by = query["operation"], query.get("column"), query.get("group_by")

    if group_by:
        grouped = data.groupby(group_by)
        result = grouped.size() if op == "count" else grouped[column].agg(op)
        result = result.sort_values(ascending=False)
        if query.get("top_n"):
            result = result.head(int(query["top_n"]))
        return result

    if op == "count":
        return len(data)
    return data[column].agg(op)


def _compare(series, cmp, val):
    return {
        "==": series == val,
        "!=": series != val,
        ">": series > val,
        ">=": series >= val,
        "<": series < val,
        "<=": series <= val,
    }[cmp]


def describe(query: dict) -> str:
    """Plain-English summary of the query, shown with every answer."""
    what = "count of rows" if query["operation"] == "count" else f"{query['operation']} of {query['column']}"
    parts = [what]
    filters = query.get("filters") or []
    if filters:
        parts.append("where " + " and ".join(f"{f['column']} {f['op']} {f['value']}" for f in filters))
    if query.get("group_by"):
        parts.append(f"by {query['group_by']}")
    if query.get("top_n"):
        parts.append(f"(top {query['top_n']})")
    return " ".join(parts)
