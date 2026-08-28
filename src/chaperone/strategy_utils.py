"""Small shared helper used by design_validation.py, build_report.py, and
review_run.py — split out to avoid a circular import between the first two
(design_validation imports classify_validation from build_report)."""


def normalize_str_list(value) -> list:
    """The validation-strategy tool schema types target_tissues/cell_types as
    list[str], but tool-call output isn't strictly schema-validated — the
    model sometimes returns a single comma-separated string instead of a
    real list, or a stringified Python list ("['a', 'b']"). Naively
    ", ".join()-ing a plain string iterates its CHARACTERS, producing
    garbage like "l, y, m, p, h, o, i, d, ..." (this actually happened for
    16/43 target_tissues and 3/43 cell_types in a real run). Normalize
    defensively: split on top-level commas only (parenthesis-aware, so
    "kidney (podocytes), lymphoid tissue" splits into 2 items not 3), and
    strip stray brackets/quotes left over from a stringified-list case."""
    if isinstance(value, list):
        return [s for v in value if (s := str(v).strip("[]'\" "))]
    if isinstance(value, str):
        parts, current, depth = [], [], 0
        for ch in value:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        parts.append("".join(current).strip())
        return [p.strip("[]'\" ") for p in parts if p.strip("[]'\" ")]
    return []
