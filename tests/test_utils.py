import pytest
from lmkg.utils import match_tool_call

# Test cases defined as dictionaries
xml_base_case = {"function_name": "get_description",
             "parameter": "entity_query",
             "value": "Michael Jordan",
             "repeat": 1,
             "expected_len": 3,
             "is_multiple": False,
             "format": "xml",
             }

xml_repeat_case = xml_base_case.copy()
xml_repeat_case["repeat"] = 2
xml_base_case["is_multiple"] = True

json_base_case = xml_base_case.copy()
json_base_case["format"] = "json"

test_cases = [xml_base_case, xml_repeat_case, json_base_case]

# Parametrized test using dictionaries
@pytest.mark.parametrize("case", test_cases)
def test_match_tool_call(case):
    function_name = case["function_name"]
    parameter = case["parameter"]
    value = case["value"]
    format = case["format"]

    if format == "xml":
        test_str = (f'<function='
                    f'{function_name}>{{"{parameter}":"{value}"}}'
                    f'</function>') * case["repeat"]
    elif format == "json":
        test_str = (f'{{"type": "function",'
                    f'"name": "{function_name}",'
                    f'"parameters": {{"{parameter}": "{value}"}}}}')
    else:
        raise ValueError(f"Unsupported format: {format}")

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
