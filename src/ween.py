import os
os.environ["HF_HOME"] = "/home/akoudri/goinfre/.cache/huggingface"
from llm_sdk import Small_LLM_Model
import json

def prompt_handle(qwen, prompt_encoded, user_request):
    prompt_part = '"prompt":'  + user_request + ','
    prompt_encoded.extend(qwen.encode(prompt_part).tolist()[0])


def function_name(qwen, prompt_encoded, functions):
    function_name = '"name":"'
    prompt_encoded.extend(qwen.encode(function_name).tolist()[0])
    allowed_functions = [f['name'] for f in functions]
    enfunctions_list = []
    for f in allowed_functions:
        encoded = qwen.encode(f).tolist()[0]
        encoded.append(497)
        if encoded not in enfunctions_list:
            enfunctions_list.append(encoded)


    position = 0
    while True:
        logits = qwen.get_logits_from_input_ids(prompt_encoded)
        chosen_ones = set()
        for enfunction in enfunctions_list:
            if position < len(enfunction):
                chosen_ones.add(enfunction[position])
        chosen_ones = list(chosen_ones)

        token_score = 0
        best_token = 0
        best_score = float('-inf')
        for token in chosen_ones:
            token_score = logits[token]
            if token_score > best_score:
                best_token = token
                best_score = token_score

        remaining = []
        for enfunction in enfunctions_list:
            try:
                exist = enfunction[position]
                if exist != best_token:
                    raise ValueError()
                remaining.append(enfunction)
            except Exception:
                pass
        enfunctions_list = remaining
        prompt_encoded.append(best_token)
        position += 1

        if len(enfunctions_list) == 1:
            the_function = enfunctions_list[0]
            prompt_encoded.extend(the_function[position:])
            break
    return the_function


def get_param_type(param, function):
    return function["parameters"][param]['type']


def param_boolean(qwen, prompt_encoded):
    expected_output = ['false', 'true']
    expected_output = [qwen.encode(e).tolist()[0][0] for e in expected_output]
    logits = qwen.get_logits_from_input_ids(prompt_encoded)

    if logits[expected_output[0]] > logits[expected_output[1]]:
        return expected_output[0], False
    else:
        return expected_output[1], True

from numpy import argmax
def param_str(qwen, prompt_encoded, tokens):

    prompt_encoded.append(tokens["quote"])

    value = ""

    for _ in range(100):
        logits = qwen.get_logits_from_input_ids(prompt_encoded)
        logits[tokens['start_quote']] = float('-inf')
        best_token = argmax(logits)
        print("best token ->", qwen.decode([best_token]))
        token_str = qwen.decode([best_token])

        if token_str.startswith('"'):
            prompt_encoded.append(tokens["quote"])
            break

        prompt_encoded.append(best_token)
        value += token_str

    return value

def param_int(qwen, prompt_encoded, tokens):
    int_vocab = tokens['digits']
    int_vocab.extend([
        tokens["minus"],
        tokens["comma"],
        tokens["end_curly"],
    ])

    value = ""

    for _ in range(20):
        logits = qwen.get_logits_from_input_ids(prompt_encoded)
        best_token = max(int_vocab, key=lambda t: logits[t])

        if best_token in (tokens["comma"], tokens["end_curly"]):
            break

        prompt_encoded.append(best_token)
        value += qwen.decode([best_token])

    return int(value)

import re
def param_float(qwen, prompt_encoded, tokens, user_request):
    has_decimal = bool(re.search(r"-?\d+\.\d+", user_request))
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

    dot_zero = qwen.encode(".0").tolist()[0]

    seen_dot = False
    value = ""

    for _ in range(20):
        logits = qwen.get_logits_from_input_ids(prompt_encoded)
        best_token = max(float_vocab, key=lambda t: logits[t])
        print("best token ->", qwen.decode([best_token]))
        if best_token == tokens["dot"]:
            seen_dot = True

        if best_token in (tokens["comma"], tokens["end_curly"]):
            if not seen_dot:
                prompt_encoded.extend(dot_zero)
                value += ".0"
            break

        prompt_encoded.append(best_token)
        value += qwen.decode([best_token])

    return float(value)
    


def params(qwen, prompt_encoded, function, user_request):

    tokens = {
        "quote_points": qwen.encode('":').tolist()[0],
        "comma": qwen.encode(', ').tolist()[0][0],
        "end_curly": qwen.encode('}').tolist()[0][0],
        "quote": qwen.encode('"').tolist()[0][0],
        "string_comma": qwen.encode('",').tolist()[0][0],
        "string_curly": qwen.encode('"}').tolist()[0][0],
        'minus': qwen.encode('-').tolist()[0][0],
        'dot': qwen.encode('.').tolist()[0][0],
        'slash_quote': qwen.encode('\\"').tolist()[0][0],
        'digits': [qwen.encode(str(i)).tolist()[0][0] for i in range(10)],
        'start_quote': qwen.encode('*"').tolist()[0][0]
    }

    generated_params = {}

    params = '"parameters": {"'
    prompt_encoded.extend(qwen.encode(params).tolist()[0])

    parameters = list(function["parameters"].keys())
    for index, param_name in enumerate(parameters):
        if index != 0:
            prompt_encoded.append(tokens["quote"])

        prompt_encoded.extend(qwen.encode(param_name).tolist()[0])
        prompt_encoded.extend(tokens["quote_points"])

        param_type = get_param_type(param_name, function)

        if param_type == "boolean":
            token, value = param_boolean(qwen, prompt_encoded)
            generated_params[param_name] = value
            prompt_encoded.append(token)

            if index < len(parameters) - 1:
                prompt_encoded.append(tokens["comma"])
            else:
                prompt_encoded.extend([tokens["end_curly"], tokens["end_curly"]])

        elif param_type == "string":
            value = param_str(qwen, prompt_encoded, tokens)
            generated_params[param_name] = value

            if index < len(parameters) - 1:
                prompt_encoded.append(tokens["string_comma"])
            else:
                prompt_encoded.extend([tokens["end_curly"], tokens["end_curly"]])

        elif param_type == "integer":
            value = param_int(qwen, prompt_encoded, tokens)
            generated_params[param_name] = value

            if index < len(parameters) - 1:
                prompt_encoded.append(tokens["comma"])
            else:
                prompt_encoded.extend([tokens["end_curly"], tokens["end_curly"]])

        elif param_type == "number" or param_type == "float":
            value = param_float(qwen, prompt_encoded, tokens, user_request)
            generated_params[param_name] = value

            if index < len(parameters) - 1:
                prompt_encoded.append(tokens["comma"])
            else:
                prompt_encoded.extend([tokens["end_curly"], tokens["end_curly"]])

    return generated_params


def generate_json(qwen, functions, user_request, prompt_encoded):

    # Generate the "prompt" field
    prompt_handle(qwen, prompt_encoded, json.dumps(user_request))

    # Generate the function name
    function_tokens = function_name(qwen, prompt_encoded, functions)
    function_name_str = qwen.decode(function_tokens[:-1])

    # Find the corresponding function definition
    function = None
    for f in functions:
        if f["name"] == function_name_str:
            function = f
            break

    if function is None:
        raise ValueError(f"Unknown function selected: {function_name_str}")

    # Generate the parameters
    generated_params = params(qwen, prompt_encoded, function, user_request)

    return {
        "prompt": user_request,
        "name": function_name_str,
        "parameters": generated_params,
    }

def dump_all(qwen, functions, prompts, arguments):

    prompt = (
        "Generate a valid JSON object with exactly three keys: prompt, name (the chosen function), and parameters. "
        "You must choose the function whose description best matches the user's request. "
        f"Available functions: {functions} "
        "Rules: "
        "- If the request says \"replace all numbers\", use regex \"[0-9]+\". "
        # "- If the request says \"replace all vowels\", use regex \"[AEIOUaeiou]\" with [] necessary. "
        # "- If the request says \"replace the word 'X'\", use regex \"X\". "
        # "- If the request says \"with NUMBERS\", replacement must be \"NUMBERS\". "
        # "- If the request says \"with asterisks\", replacement must be \"*\". "
        # "- If the request says \"with dog\", replacement must be \"dog\". "
        # "Example output: "
        # "{\"prompt\":\"Add 2 and 8\",\"name\":\"fn_add_numbers\",\"parameters\":{\"a\":2,\"b\":8}} "
        # "Example 2: "
        # "{\"prompt\":\"Replace all numbers in 'Hello 34 I\\\"m 233 years old' with NUMBERS\","
        # "\"name\":\"fn_substitute_string_with_regex\","
        # "\"parameters\":{"
        # "\"source_string\":\"Hello 34 I'm 233 years old\","
        # "\"regex\":\"[0-9]+\","
        # "\"replacement\":\"NUMBERS\""
        # "}} "
        # "Example 3: "
        # "{\"prompt\":\"Replace all vowels in 'Programming is fun' with asterisks\","
        # "\"name\":\"fn_substitute_string_with_regex\","
        # "\"parameters\":{"
        # "\"source_string\":\"Programming is fun\","
        # "\"regex\":\"[AEIOUaeiou]\","
        # "\"replacement\":\"*\""
        # "}} "
        # "Example 4:"
        # "{\"name\":\"fn_substitute_string_with_regex\",\"parameters\":{\"source_string\":\"The cat sat on the mat with another cat\",\"regex\":\"cat\",\"replacement\":\"dog\"}}"
        # "Answer: {"
        "Example output: "
        "{\"prompt\":\"Compute the sum of 15 and 27\",\"name\":\"fn_add_numbers\",\"parameters\":{\"a\":15,\"b\":27}} "

        "Example 2: "
        "{\"prompt\":\"Replace every sequence of digits in 'Order 512 costs 49 dollars' with <NUM>\","
        "\"name\":\"fn_substitute_string_with_regex\","
        "\"parameters\":{"
        "\"source_string\":\"Order 512 costs 49 dollars\","
        "\"regex\":\"[0-9]+\","
        "\"replacement\":\"<NUM>\""
        "}} "

        "Example 3: "
        "{\"prompt\":\"Replace every vowel in 'Artificial Intelligence' with *\","
        "\"name\":\"fn_substitute_string_with_regex\","
        "\"parameters\":{"
        "\"source_string\":\"Artificial Intelligence\","
        "\"regex\":\"[AEIOUaeiou]\","
        "\"replacement\":\"*\""
        "}} "

        "Example 4: "
        "{\"prompt\":\"Replace every occurrence of 'apple' with 'orange' in 'apple pie and apple juice'\","
        "\"name\":\"fn_substitute_string_with_regex\","
        "\"parameters\":{"
        "\"source_string\":\"apple pie and apple juice\","
        "\"regex\":\"apple\","
        "\"replacement\":\"orange\""
        "}} "

        "Answer: {"
            )

    base_prompt_encoded = qwen.encode(prompt).tolist()[0]

    if isinstance(prompts, dict):
        prompts = [prompts]

    results = []

    for item in prompts:
        user_request = item["prompt"]

        # Every generation starts from the same base context
        prompt_encoded = base_prompt_encoded.copy()

        generated = generate_json(
            qwen,
            functions,
            user_request,
            prompt_encoded
        )

        results.append(generated)

    with open(arguments.output, "w") as f:
        json.dump(results, f, indent=4)