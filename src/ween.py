import json
import re
from typing import Any, cast
from llm_sdk import Small_LLM_Model  # type: ignore
from numpy import argmax


# Simple type aliases (replace TypedDict/Protocol runtime definitions to keep mypy quiet)
ParameterDef = dict[str, Any]
FunctionDef = dict[str, Any]
TokenDict = dict[str, Any]
GeneratedOutput = dict[str, Any]
DumpArguments = Any


def prompt_handle(
    qwen: Small_LLM_Model,
    prompt_encoded: list[int],
    user_request: str
) -> None:
    """Handles the prompt generation by appending the user request.

    Args:
        qwen (Small_LLM_Model): The LLM model instance.
        prompt_encoded (list[int]): The list of encoded tokens.
        user_request (str): The user's prompt request.
    """
    prompt_part: str = '"prompt":' + user_request + ","
    encoded_part: list[int] = [int(x) for x in qwen.encode(prompt_part)[0]]
    prompt_encoded.extend(encoded_part)


def function_name(
    qwen: Small_LLM_Model,
    prompt_encoded: list[int],
    functions: list[FunctionDef]
) -> list[int]:
    """Determines the target function name and appends it to the prompt.

    Args:
        qwen (Small_LLM_Model): The LLM model instance.
        prompt_encoded (list[int]): The list of encoded tokens.
        functions (list[FunctionDef]): A list of allowed function dictionaries.

    Returns:
        list[int]: The token IDs forming the chosen function name.
    """
    function_name_str: str = '"name":"'
    fn_tokens: list[int] = [int(x) for x in qwen.encode(function_name_str)[0]]
    prompt_encoded.extend(fn_tokens)
    allowed_functions: list[str] = [f["name"] for f in functions]
    encoded_functions: list[list[int]] = []

    for f in allowed_functions:
        encoded: list[int] = [int(x) for x in qwen.encode(f)[0]]
        encoded.append(497)  # sentinel token appended to mark end
        if encoded not in encoded_functions:
            encoded_functions.append(encoded)

    position: int = 0
    function_tokens: list[int] = []

    while True:
        logits: list[float] = cast(
            list[float], qwen.get_logits_from_input_ids(prompt_encoded)
        )
        chosen_set: set[int] = set()

        for enc in encoded_functions:
            if position < len(enc):
                chosen_set.add(enc[position])

        chosen_ones: list[int] = list(chosen_set)

        best_token: int = 0
        best_score: float = float("-inf")

        for token in chosen_ones:
            # guard against potential short logits vector
            if token >= len(logits):
                continue
            token_score = float(logits[token])
            if token_score > best_score:
                best_token = token
                best_score = token_score

        remaining: list[list[int]] = []
        for enc in encoded_functions:
            try:
                exist: int = enc[position]
                if exist != best_token:
                    raise ValueError()
                remaining.append(enc)
            except Exception:
                pass

        encoded_functions = remaining
        prompt_encoded.append(best_token)
        position += 1

        if len(encoded_functions) == 1:
            function_tokens = encoded_functions[0]
            prompt_encoded.extend(function_tokens[position:])
            break

    return function_tokens


def get_param_type(param: str, function: FunctionDef) -> str:
    """Retrieves and normalizes the type of a specific parameter.

    Args:
        param (str): The name of the parameter.
        function (FunctionDef): The function definition dictionary.

    Returns:
        str: The normalized string type of the parameter.
    """
    param_def: ParameterDef = cast(
        ParameterDef, function["parameters"][param]
    )
    return param_def["type"].strip().lower()


def param_boolean(
    qwen: Small_LLM_Model, prompt_encoded: list[int]
) -> tuple[int, bool]:
    """Generates a boolean parameter value based on the model's logits."""
    expected_strings: list[str] = ["false", "true"]
    expected_tokens: list[int] = [
        [int(x) for x in qwen.encode(e)[0]][0] for e in expected_strings
    ]
    logits: list[float] = cast(
        list[float], qwen.get_logits_from_input_ids(prompt_encoded)
    )

    # guard indices
    t0, t1 = expected_tokens[0], expected_tokens[1]
    v0 = logits[t0] if t0 < len(logits) else float("-inf")
    v1 = logits[t1] if t1 < len(logits) else float("-inf")

    if v0 > v1:
        return t0, False
    return t1, True


def param_str(
    qwen: Small_LLM_Model, prompt_encoded: list[int], tokens: TokenDict
) -> str:
    """Generates a string parameter value based on the model's logits."""
    prompt_encoded.append(tokens["quote"])  # opening quote
    value: str = ""

    for _ in range(30):
        logits: list[float] = cast(
            list[float], qwen.get_logits_from_input_ids(prompt_encoded)
        )
        # avoid selecting the opening quote token again
        if (
            tokens.get("start_quote") is not None
            and tokens["start_quote"] < len(logits)
        ):
            logits[tokens["start_quote"]] = float("-inf")
        best_token: int = int(argmax(logits))

        decoded: str = cast(str, qwen.decode([best_token]))
        print("best token ->", decoded)

        if decoded.startswith('"'):
            prompt_encoded.append(tokens["quote"])  # closing quote
            break

        prompt_encoded.append(best_token)
        value += decoded

    return value


def param_int(
    qwen: Small_LLM_Model, prompt_encoded: list[int], tokens: TokenDict
) -> int:
    """Generates an integer parameter value based on the model's logits."""
    int_vocab: list[int] = list(tokens["digits"])  # copy to avoid mutating
    int_vocab.extend(
        [tokens["minus"], tokens["comma"], tokens["end_curly"]]
    )

    value: str = ""

    for _ in range(30):
        logits: list[float] = cast(
            list[float], qwen.get_logits_from_input_ids(prompt_encoded)
        )
        best_token: int = max(
            int_vocab,
            key=lambda t: (
                logits[t] if t < len(logits) else float("-inf")
            ),
        )

        if best_token in (tokens["comma"], tokens["end_curly"]):
            break

        prompt_encoded.append(best_token)
        value += cast(str, qwen.decode([best_token]))

    value = value.strip()
    if not value:
        raise ValueError("No digits generated for integer parameter")

    return int(value)


def param_float(
    qwen: Small_LLM_Model,
    prompt_encoded: list[int],
    tokens: TokenDict,
    user_request: str
) -> float:
    """Generates a float parameter value based on the model's logits."""
    has_decimal: bool = bool(re.search(r"-?\d+\.\d+", user_request))

    if has_decimal:
        float_vocab: list[int] = [
            *tokens["digits"],
            tokens["minus"],
            tokens["dot"],
            tokens["comma"],
            tokens["end_curly"],
        ]
    else:
        float_vocab = [
            *tokens["digits"],
            tokens["minus"],
            tokens["comma"],
            tokens["end_curly"],
        ]

    dot_zero: list[int] = [int(x) for x in qwen.encode(".0")[0]]

    seen_dot: bool = False
    value: str = ""

    for _ in range(30):
        logits: list[float] = cast(
            list[float], qwen.get_logits_from_input_ids(prompt_encoded)
        )
        best_token: int = max(
            float_vocab,
            key=lambda t: (
                logits[t] if t < len(logits) else float("-inf")
            ),
        )

        decoded: str = cast(str, qwen.decode([best_token]))
        print("best token ->", decoded)

        if best_token == tokens["dot"]:
            seen_dot = True

        if best_token in (tokens["comma"], tokens["end_curly"]):
            if not seen_dot:
                prompt_encoded.extend(dot_zero)
                value += ".0"
            break

        prompt_encoded.append(best_token)
        value += decoded

    value = value.strip()
    if not value:
        raise ValueError("No digits generated for float parameter")

    return float(value)


def check_params_exist(function: FunctionDef) -> bool:
    """Checks whether the specified function possesses any parameters."""
    if not function.get("parameters"):
        return False
    return True


def params(
    qwen: Small_LLM_Model,
    prompt_encoded: list[int],
    function: FunctionDef,
    user_request: str
) -> dict[str, Any] | bool:
    """Evaluates and generates all required parameters for a function."""
    if not check_params_exist(function):
        params_str: str = '"parameters": {}'
        prompt_encoded.extend([int(x) for x in qwen.encode(params_str)[0]])
        return False

    tokens: TokenDict = {
        "quote_points": [int(x) for x in qwen.encode('\":')[0]],
        "comma": [int(x) for x in qwen.encode(', ')[0]][0],
        "end_curly": [int(x) for x in qwen.encode('}')[0]][0],
        "quote": [int(x) for x in qwen.encode('"')[0]][0],
        "string_comma": [int(x) for x in qwen.encode('",')[0]][0],
        "string_curly": [int(x) for x in qwen.encode('\"}')[0]][0],
        "minus": [int(x) for x in qwen.encode('-')[0]][0],
        "dot": [int(x) for x in qwen.encode('.')[0]][0],
        "slash_quote": [int(x) for x in qwen.encode('\\"')[0]][0],
        "digits": [
            [int(x) for x in qwen.encode(str(i))[0]][0] for i in range(10)
        ],
        "start_quote": [int(x) for x in qwen.encode('*"')[0]][0],
    }

    generated_params: dict[str, Any] = {}

    params_prefix: str = '"parameters": {"'
    prompt_encoded.extend([int(x) for x in qwen.encode(params_prefix)[0]])

    parameters: list[str] = list(function.get("parameters", {}).keys())
    for index, param_name in enumerate(parameters):
        if index != 0:
            prompt_encoded.append(tokens["quote"])

        param_tokens: list[int] = [int(x) for x in qwen.encode(param_name)[0]]
        prompt_encoded.extend(param_tokens)
        prompt_encoded.extend(tokens["quote_points"])

        param_type: str = get_param_type(param_name, function)

        if param_type == "boolean":
            token, bool_value = param_boolean(qwen, prompt_encoded)
            generated_params[param_name] = bool_value
            prompt_encoded.append(token)

            if index < len(parameters) - 1:
                prompt_encoded.append(tokens["comma"])
            else:
                prompt_encoded.extend(
                    [tokens["end_curly"], tokens["end_curly"]]
                )

        elif param_type == "string":
            str_value: str = param_str(qwen, prompt_encoded, tokens)
            generated_params[param_name] = str_value

            if index < len(parameters) - 1:
                prompt_encoded.append(tokens["string_comma"])
            else:
                prompt_encoded.extend(
                    [tokens["end_curly"], tokens["end_curly"]]
                )

        elif param_type == "integer":
            int_value: int = param_int(qwen, prompt_encoded, tokens)
            generated_params[param_name] = int_value

            if index < len(parameters) - 1:
                prompt_encoded.append(tokens["comma"])
            else:
                prompt_encoded.extend(
                    [tokens["end_curly"], tokens["end_curly"]]
                )

        elif param_type in ("number", "float"):
            float_value: float = param_float(
                qwen, prompt_encoded, tokens, user_request
            )
            generated_params[param_name] = float_value

            if index < len(parameters) - 1:
                prompt_encoded.append(tokens["comma"])
            else:
                prompt_encoded.extend(
                    [tokens["end_curly"], tokens["end_curly"]]
                )

    return generated_params


def generate_json(
    qwen: Small_LLM_Model,
    functions: list[FunctionDef],
    user_request: str,
    prompt_encoded: list[int]
) -> GeneratedOutput:
    """Orchestrates generation to produce a structured JSON object."""
    # Generate the "prompt" field
    prompt_handle(qwen, prompt_encoded, json.dumps(user_request))

    # Generate the function name
    function_tokens: list[int] = function_name(
        qwen, prompt_encoded, functions
    )
    function_name_str: str = cast(str, qwen.decode(function_tokens[:-1]))

    # Find the corresponding function definition
    function: FunctionDef | None = None
    for f in functions:
        if f["name"] == function_name_str:
            function = f
            break

    if function is None:
        raise ValueError(f"Unknown function selected: {function_name_str}")

    # Generate the parameters
    generated_params_res: dict[str, Any] | bool = params(
        qwen, prompt_encoded, function, user_request
    )
    generated_params: dict[str, Any] = {}
    if not generated_params_res:
        generated_params = {}
    else:
        generated_params = cast(dict[str, Any], generated_params_res)

    return {
        "prompt": user_request,
        "name": function_name_str,
        "parameters": generated_params,
    }


def dump_all(
    qwen: Small_LLM_Model,
    functions: list[FunctionDef] | FunctionDef,
    prompts: list[dict[str, str]] | dict[str, str],
    arguments: DumpArguments
) -> None:
    """Processes multiple prompts with the LLM and dumps output to a file."""
    prompt_lines: list[str] = [
        "Generate a valid JSON object with exactly three keys: prompt, ",
        "name (the chosen function), and parameters. ",
        "You must choose the function whose description best matches the ",
        "user's request. ",
        f"Available functions: {functions} ",
        "Rules: ",
        '- If the request says "replace all numbers", use regex "[0-9]+". ',
        "Example output: ",
        '{"prompt":"Compute the sum of 15 and 27",',
        '"name":"fn_add_numbers",',
        '"parameters":{"a":15,"b":27}} ',
        "Example 2: ",
        '{"prompt":"Replace every sequence of digits in \'Order 512 ',
        "costs 49 dollars' with <NUM>",
        ',"name":"fn_substitute_string_with_regex",',
        '"parameters":{',
        '"source_string":"Order 512 costs 49 dollars",',
        '"regex":"[0-9]+",',
        '"replacement":"<NUM>"',
        '}} ',
        "Example 3: ",
        '{"prompt":"Replace every vowel in \'Artificial Intelligence\' ',
        '"with *",',
        '"name":"fn_substitute_string_with_regex",',
        '"parameters":{',
        '"source_string":"Artificial Intelligence",',
        '"regex":"[AEIOUaeiou]",',
        '"replacement":"*"',
        '}} ',
        "Example 4: ",
        '{"prompt":"Replace every occurrence of \'apple\' with \'orange\' ',
        "in 'apple pie and apple juice'",',
        '"name":"fn_substitute_string_with_regex",',
        '"parameters":{',
        '"source_string":"apple pie and apple juice",',
        '"regex":"apple",',
        '"replacement":"orange"',
        '}} ',
        "Answer: {",
    ]

    prompt: str = "".join(prompt_lines)

    base_prompt_encoded: list[int] = [
        int(x) for x in qwen.encode(prompt)[0]
    ]

    prompts_list: list[dict[str, str]] = (
        [prompts] if isinstance(prompts, dict) else prompts
    )
    functions_list: list[FunctionDef] = (
        [functions] if isinstance(functions, dict) else functions
    )

    results: list[GeneratedOutput] = []

    for item in prompts_list:
        key: str = list(item.keys())[0]
        user_request: str = item[key]

        # Every generation starts from the same base context
        prompt_encoded: list[int] = base_prompt_encoded.copy()

        generated: GeneratedOutput = generate_json(
            qwen, functions_list, user_request, prompt_encoded
        )

        results.append(generated)

    with open(arguments.output, "w") as f:
        json.dump(results, f, indent=4)
