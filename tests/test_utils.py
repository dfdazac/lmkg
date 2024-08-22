import pytest
from lmkg.utils import match_tool_call

# Test cases defined as dictionaries
test_cases = [
    {
        "function_name": "get_description",
        "parameter": "entity_query",
        "value": "Michael Jordan",
        "repeat": 1,
        "expected_len": 3,
        "is_multiple": False,
    },
    {
        "function_name": "get_description",
        "parameter": "entity_query",
        "value": "Michael Jordan",
        "repeat": 2,
        "expected_len": 3,
        "is_multiple": True,
    },
]

# Parametrized test using dictionaries
@pytest.mark.parametrize("case", test_cases)
def test_match_tool_call(case):
    function_name = case["function_name"]
    parameter = case["parameter"]
    value = case["value"]

    test_str = (f'<function='
                f'{function_name}>{{"{parameter}":"{value}"}}'
                f'</function>') * case["repeat"]

    result = match_tool_call(test_str)

    # Assertions common to both test cases
    assert len(result) == case["expected_len"]
    assert result[0] == function_name
    assert result[1] == {parameter: value}

    # Assertions specific to single or multiple function calls
    if case["is_multiple"]:
        assert result[2] is not None
    else:
        assert result[2] is None
