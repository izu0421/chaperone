"""Regression tests for normalize_str_list — guards against the exact bug
that shipped a real run: ", ".join()-ing a string the model returned instead
of a list iterates its characters, producing "l, y, m, p, h, o, i, d, ...".
"""
from chaperone.strategy_utils import normalize_str_list


def test_already_a_clean_list_passes_through():
    assert normalize_str_list(["bone marrow", "lymphoid tissue"]) == ["bone marrow", "lymphoid tissue"]


def test_comma_separated_string_is_split():
    assert normalize_str_list("bone marrow, lymphoid tissue") == ["bone marrow", "lymphoid tissue"]


def test_parenthesis_aware_split_does_not_split_inside_parens():
    value = "kidney (podocytes), lymphoid tissue"
    assert normalize_str_list(value) == ["kidney (podocytes)", "lymphoid tissue"]


def test_stringified_python_list_strips_brackets_and_quotes():
    value = "['cerebral cortex', 'brain (cortical neurons)', 'cerebellum (Bergmann glia)']"
    assert normalize_str_list(value) == ["cerebral cortex", "brain (cortical neurons)", "cerebellum (Bergmann glia)"]


def test_list_with_bracket_quote_noise_in_items_is_cleaned():
    # what a naive comma-split of a stringified list looks like before this
    # function existed — confirms the LIST branch also strips stray chars
    value = ["['cerebral cortex'", "'brain (cortical neurons)'", "'cerebellum (Bergmann glia)']"]
    assert normalize_str_list(value) == ["cerebral cortex", "brain (cortical neurons)", "cerebellum (Bergmann glia)"]


def test_none_and_empty_return_empty_list():
    assert normalize_str_list(None) == []
    assert normalize_str_list("") == []
    assert normalize_str_list([]) == []


def test_does_not_corrupt_into_characters():
    # the actual historical bug: ", ".join("lymphoid tissue") -> "l, y, m, ..."
    result = normalize_str_list("lymphoid tissue")
    assert result == ["lymphoid tissue"]
    assert "l" not in result  # would be a single-char item if it had regressed
