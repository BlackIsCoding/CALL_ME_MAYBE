import json
import re
from typing import Any, Protocol, TypedDict, cast, List
from llm_sdk import Small_LLM_Model  # type: ignore
from numpy import argmax


class ParameterDef(TypedDict):
    """Represents a parameter definition inside a function's schema."""
    type: str


class FunctionDef(TypedDict):
    """Represents an available function definition."""
    name: str
    parameters: dict[str, Any]


class TokenDict(TypedDict):
    """Stores necessary token sequences and single token IDs."""
    quote_points: List[int]
    comma: int
    end_curly: int
    quote: int
    string_comma: int
    string_curly: int
    minus: int
    dot: int
    slash_quote: int
    digits: List[int]
    start_quote: int


class GeneratedOutput(TypedDict):
    """Represents the final formatted output for the LLM."""
    prompt: str
    name: str
    parameters: dict[str, Any]


class DumpArguments(Protocol):
    """Protocol for the arguments object expected by dump_all."""
    output: str


def prompt_handle(
    qwen: Small_LLM_Model,
    prompt_encoded: List[int],
    user_request: str
) -> None:
    """Handles the prompt generation by appending the user request.

    Args:
        qwen (Small_LLM_Model): The LLM model instance.
        prompt_encoded (List[int]): The List of encoded tokens.
        user_request (str): The user's prompt request.

    Returns:
        None
    """
    prompt_part: str = '"prompt":' + user_request + ","
    encoded_part: List[int] = [int(x) for x in qwen.encode(prompt_part)[0]]
    prompt_encoded.extend(encoded_part)


def function_name(
    qwen: Small_LLM_Model,
    prompt_encoded: List[int],
    functions: List[FunctionDef]
) -> List[int]:
    """Determines the target function name and appends it to the prompt.

    Args:
        qwen (Small_LLM_Model): The LLM model instance.
        prompt_encoded (List[int]): The List of encoded tokens.
        functions (List[FunctionDef]): A List of allowed function dictionaries.

    Returns:
        List[int]: The token IDs forming the chosen function name.

    Raises:
        ValueError: Re-raises implicitly internally if matching fails.
    """
    function_name_str: str = '"name":"'
    fn_tokens: List[int] = [int(x) for x in qwen.encode(function_name_str)[0]]
    prompt_encoded.extend(fn_tokens)
    allowed_functions: List[str] = [f["name"] for f in functions]
    enfunctions_List: List[List[int]] = []

    for f in allowed_functions:
        encoded: List[int] = [int(x) for x in qwen.encode(f)[0]]
        encoded.append(497)
        if encoded not in enfunctions_List:
            enfunctions_List.append(encoded)

    position: int = 0
    the_function: List[int] = []

    while True:
        logits: List[float] = cast(
            List[float], qwen.get_logits_from_input_ids(prompt_encoded)
        )
        chosen_ones_set: set[int] = set()

        for enfunction in enfunctions_List:
            if position < len(enfunction):
                chosen_ones_set.add(enfunction[position])

        chosen_ones: List[int] = list(chosen_ones_set)

        token_score: float = 0.0
        best_token: int = 0
        best_score: float = float("-inf")

        for token in chosen_ones:
            token_score = float(logits[token])
            if token_score > best_score:
                best_token = token
                best_score = token_score

        remaining: List[List[int]] = []
        for enfunction in enfunctions_List:
            try:
                exist: int = enfunction[position]
                if exist != best_token:
                    raise ValueError()
                remaining.append(enfunction)
            except Exception:
                pass

        enfunctions_List = remaining
        prompt_encoded.append(best_token)
        position += 1

        if len(enfunctions_List) == 1:
            the_function = enfunctions_List[0]
            prompt_encoded.extend(the_function[position:])
            break

    return the_function


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
    qwen: Small_LLM_Model, prompt_encoded: List[int]
) -> tuple[int, bool]:
    """Generates a boolean parameter value based on the model's logits.

    Args:
        qwen (Small_LLM_Model): The LLM model instance.
        prompt_encoded (List[int]): The sequence of encoded tokens.

    Returns:
        tuple[int, bool]: Token ID and the respective boolean value.
    """
    expected_strings: List[str] = ["false", "true"]
    expected_tokens: List[int] = [[int(x) for x in qwen.encode(e)[0]][0] for e in expected_strings]
    logits: List[float] = cast(
        List[float], qwen.get_logits_from_input_ids(prompt_encoded)
    )

    if logits[expected_tokens[0]] > logits[expected_tokens[1]]:
        return expected_tokens[0], False
    else:
        return expected_tokens[1], True


def param_str(
    qwen: Small_LLM_Model, prompt_encoded: List[int], tokens: TokenDict
) -> str:
    """Generates a string parameter value based on the model's logits.

    Args:
        qwen (Small_LLM_Model): The LLM model instance.
        prompt_encoded (List[int]): The sequence of encoded tokens.
        tokens (TokenDict): A dictionary containing token vocabulary.

    Returns:
        str: The generated string value.
    """
    prompt_encoded.append(tokens["quote"])
    value: str = ""

    for _ in range(30):
        logits: List[float] = cast(
            List[float], qwen.get_logits_from_input_ids(prompt_encoded)
        )
        logits[tokens["start_quote"]] = float("-inf")
        best_token: int = int(argmax(logits))

        decoded: str = cast(str, qwen.decode([best_token]))
        print("best token ->", decoded)
        token_str: str = decoded

        if token_str.startswith('"'):
            prompt_encoded.append(tokens["quote"])
            break

        prompt_encoded.append(best_token)
        value += token_str

    return value


def param_int(
    qwen: Small_LLM_Model, prompt_encoded: List[int], tokens: TokenDict
) -> int:
    """Generates an integer parameter value based on the model's logits.

    Args:
        qwen (Small_LLM_Model): The LLM model instance.
        prompt_encoded (List[int]): The sequence of encoded tokens.
        tokens (TokenDict): A dictionary containing token vocabulary.

    Returns:
        int: The generated integer value.
    """
    int_vocab: List[int] = tokens["digits"]
    int_vocab.extend(
        [
            tokens["minus"],
            tokens["comma"],
            tokens["end_curly"],
        ]
    )

    value: str = ""

    for _ in range(30):
        logits: List[float] = cast(
            List[float], qwen.get_logits_from_input_ids(prompt_encoded)
        )
        best_token: int = max(int_vocab, key=lambda t: logits[t])

        if best_token in (tokens["comma"], tokens["end_curly"]):
            break

        prompt_encoded.append(best_token)
        value += cast(str, qwen.decode([best_token]))

    return int(value)


def param_float(
    qwen: Small_LLM_Model,
    prompt_encoded: List[int],
    tokens: TokenDict,
    user_request: str
) -> float:
    """Generates a float parameter value based on the model's logits.

    Args:
        qwen (Small_LLM_Model): The LLM model instance.
        prompt_encoded (List[int]): The sequence of encoded tokens.
        tokens (TokenDict): A dictionary containing token vocabulary.
        user_request (str): The user's prompt request.

    Returns:
        float: The generated float value.
    """
    has_decimal: bool = bool(re.search(r"-?\d+\.\d+", user_request))
    float_vocab: List[int]

    if has_decimal:
        float_vocab = [
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

    dot_zero: List[int] = [int(x) for x in qwen.encode(".0")[0]]

    seen_dot: bool = False
    value: str = ""

    for _ in range(30):
        logits: List[float] = cast(
            List[float], qwen.get_logits_from_input_ids(prompt_encoded)
        )
        best_token: int = max(float_vocab, key=lambda t: logits[t])

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

    return float(value)


def check_params_exist(function: FunctionDef) -> bool:
    """Checks whether the specified function possesses any parameters.

    Args:
        function (FunctionDef): The function definition dictionary.

    Returns:
        bool: True if parameters exist and are not empty, False otherwise.
    """
    if not function["parameters"]:
        return False
    return True


def params(
    qwen: Small_LLM_Model,
    prompt_encoded: List[int],
    function: FunctionDef,
    user_request: str
) -> dict[str, Any] | bool:
    """Evaluates and generates all required parameters for a function.

    Args:
        qwen (Small_LLM_Model): The LLM model instance.
        prompt_encoded (List[int]): The List of encoded tokens.
        function (FunctionDef): The function schema.
        user_request (str): The original user prompt request.

    Returns:
        dict[str, Any] | bool: Populated parameters or False if none exist.
    """
    if not check_params_exist(function):
        params_str: str = '"parameters": {}'
        prompt_encoded.extend([int(x) for x in qwen.encode(params_str)[0]])
        return False

    tokens: TokenDict = {
        "quote_points": [int(x) for x in qwen.encode('":')[0]],
        "comma": [int(x) for x in qwen.encode(', ')[0]][0],
        "end_curly": [int(x) for x in qwen.encode('}')[0]][0],
        "quote": [int(x) for x in qwen.encode('"')[0]][0],
        "string_comma": [int(x) for x in qwen.encode('",')[0]][0],
        "string_curly": [int(x) for x in qwen.encode('"}')[0]][0],
        "minus": [int(x) for x in qwen.encode('-')[0]][0],
        "dot": [int(x) for x in qwen.encode('.')[0]][0],
        "slash_quote": [int(x) for x in qwen.encode('\\"')[0]][0],
        "digits": [ [int(x) for x in qwen.encode(str(i))[0]][0] for i in range(10) ],
        "start_quote": [int(x) for x in qwen.encode('*"')[0]][0],
    }

    generated_params: dict[str, Any] = {}

    params_prefix: str = '"parameters": {"'
    prompt_encoded.extend([int(x) for x in qwen.encode(params_prefix)[0]])

    parameters: List[str] = list(function["parameters"].keys())
    for index, param_name in enumerate(parameters):
        if index != 0:
            prompt_encoded.append(tokens["quote"])

        param_tokens: List[int] = [int(x) for x in qwen.encode(param_name)[0]]
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

        elif param_type == "number" or param_type == "float":
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
    functions: List[FunctionDef],
    user_request: str,
    prompt_encoded: List[int]
) -> GeneratedOutput:
    """Orchestrates generation to produce a structured JSON object.

    Args:
        qwen (Small_LLM_Model): The LLM model instance.
        functions (List[FunctionDef]): Allowed function schema.
        user_request (str): The raw string request.
        prompt_encoded (List[int]): Sequence of encoded context tokens.

    Returns:
        GeneratedOutput: Structurally valid parsed generated dict.

    Raises:
        ValueError: If an unknown function is selected during generation.
    """
    # Generate the "prompt" field
    prompt_handle(qwen, prompt_encoded, json.dumps(user_request))

    # Generate the function name
    function_tokens: List[int] = function_name(qwen, prompt_encoded, functions)
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
    functions: List[FunctionDef] | FunctionDef,
    prompts: List[dict[str, str]] | dict[str, str],
    arguments: DumpArguments
) -> None:
    """Processes multiple prompts with the LLM and dumps output to a file.

    Args:
        qwen (Small_LLM_Model): The LLM model instance.
        functions (List[FunctionDef] | FunctionDef): Schemas of functions.
        prompts (List[dict[str, str]] | dict[str, str]): Batch data requests.
        arguments (DumpArguments): Runtime flags for paths/configurations.

    Returns:
        None
    """
    prompt: str = (
        "Generate a valid JSON object with exactly three keys: prompt, "
        "name (the chosen function), and parameters. "
        "You must choose the function whose description best matches the "
        "user's request. "
        f"Available functions: {functions} "
        "Rules: "
        '- If the request says "replace all numbers", use regex "[0-9]+". '
        "Example output: "
        '{"prompt":"Compute the sum of 15 and 27",'
        '"name":"fn_add_numbers",'
        '"parameters":{"a":15,"b":27}} '
        "Example 2: "
        "{\"prompt\":\"Replace every sequence of digits in 'Order 512 "
        "costs 49 dollars' with <NUM>\","
        '"name":"fn_substitute_string_with_regex",'
        '"parameters":{'
        '"source_string":"Order 512 costs 49 dollars",'
        '"regex":"[0-9]+",'
        '"replacement":"<NUM>"'
        "}} "
        "Example 3: "
        "{\"prompt\":\"Replace every vowel in 'Artificial Intelligence' "
        "with *\","
        '"name":"fn_substitute_string_with_regex",'
        '"parameters":{'
        '"source_string":"Artificial Intelligence",'
        '"regex":"[AEIOUaeiou]",'
        '"replacement":"*"'
        "}} "
        "Example 4: "
        "{\"prompt\":\"Replace every occurrence of 'apple' with 'orange' "
        "in 'apple pie and apple juice'\","
        '"name":"fn_substitute_string_with_regex",'
        '"parameters":{'
        '"source_string":"apple pie and apple juice",'
        '"regex":"apple",'
        '"replacement":"orange"'
        "}} "
        "Answer: {"
    )

    base_prompt_encoded: List[int] = [int(x) for x in qwen.encode(prompt)[0]]

    prompts_List: List[dict[str, str]] = (
        [prompts] if isinstance(prompts, dict) else prompts
    )
    functions_List: List[FunctionDef] = (
        [functions] if isinstance(functions, dict) else functions
    )

    results: List[GeneratedOutput] = []

    for item in prompts_List:
        key: str = list(item.keys())[0]
        user_request: str = item[key]

        # Every generation starts from the same base context
        prompt_encoded: List[int] = base_prompt_encoded.copy()

        generated: GeneratedOutput = generate_json(
            qwen,
            functions_List,
            user_request,
            prompt_encoded
        )

        results.append(generated)

    with open(arguments.output, "w") as f:
        json.dump(results, f, indent=4)
