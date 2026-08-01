import os
os.environ["HF_HOME"] = "/home/akoudri/goinfre/.cache/huggingface"
from llm_sdk import Small_LLM_Model
from numpy import argmax
import json



def get_vocab_str(qwen):

    vocab_list = []
    with open(qwen.get_path_to_vocab_file(), 'r') as f:
        vocab = json.load(f)
        for token_id, token in vocab.items():
            charachter = qwen.decode(token)
            if charachter == '\\"':
                vocab_list.append(token)
                continue
            if '"' in charachter or '}' in charachter:
                continue
            elif "\\[" in charachter or "\\]" in charachter:
                continue
            else:
                vocab_list.append(token)
    return vocab_list


def get_vocab_number(qwen):

    vocab_list = []
    with open(qwen.get_path_to_vocab_file(), 'r') as f:
        vocab = json.load(f)
        for token_id, token in vocab.items():
            charachter = qwen.decode(token)
            if charachter and all(ch.isdigit() for ch in charachter):
                vocab_list.append(token)
    return vocab_list



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
        return expected_output[0]
    else:
        return expected_output[1]


def param_str(qwen, prompt_encoded, user_request, tokens):

    prompt_encoded.append(tokens["quote"])
    allowed_tokens = get_vocab_str(qwen)
    allowed_tokens.extend([tokens['quote']])

    for _ in range(20):
        logits = qwen.get_logits_from_input_ids(prompt_encoded)

        best_score = float("-inf")
        best_token = None
        for token in allowed_tokens:
            if logits[token] > best_score:
                best_score = logits[token]
                best_token = token

        prompt_encoded.append(best_token)
        if best_token == tokens['quote']:
            break

def param_int(qwen, prompt_encoded, user_request, tokens):
    int_vocab = get_vocab_number(qwen)
    int_vocab.extend([tokens['comma'],tokens['end_curly'], tokens['minus']])

    for _ in range(20):
        logits = qwen.get_logits_from_input_ids(prompt_encoded)

        best_score = float("-inf")
        best_token = None
        for token in int_vocab:
            if logits[token] > best_score:
                best_score = logits[token]
                best_token = token
        if best_token in (tokens['comma'], tokens['end_curly']):
            break
        prompt_encoded.append(best_token)

def param_float(qwen, prompt_encoded, tokens):
    float_vocab = get_vocab_number(qwen)
    float_vocab.extend([
        tokens["dot"],
        tokens["minus"],
        tokens["comma"],
        tokens["end_curly"],
    ])

    seen_dot = False

    for _ in range(20):
        logits = qwen.get_logits_from_input_ids(prompt_encoded)

        best_score = float("-inf")
        best_token = None

        for token in float_vocab:
            if logits[token] > best_score:
                best_score = logits[token]
                best_token = token

        if best_token == tokens["dot"]:
            seen_dot = True

        if best_token in (tokens["comma"], tokens["end_curly"]):
            if not seen_dot:
                prompt_encoded.append(tokens["dot"])
                prompt_encoded.append(qwen.encode("0").tolist()[0][0])
            break

        prompt_encoded.append(best_token)
    


def params(qwen, prompt_encoded, function, user_request):

    tokens = {
        "quote_points": qwen.encode('":').tolist()[0],
        "comma": qwen.encode(', ').tolist()[0][0],
        "end_curly": qwen.encode('}').tolist()[0][0],
        "quote": qwen.encode('"').tolist()[0][0],
        "string_comma": qwen.encode('",').tolist()[0][0],
        "string_curly": qwen.encode('"}').tolist()[0][0],
        'minus': qwen.encode('-').tolist()[0][0],
        'dot': qwen.encode('.').tolist()[0][0]
    }

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
            prompt_encoded.append(param_boolean(qwen, prompt_encoded))

            if index < len(parameters) - 1:
                prompt_encoded.append(tokens["comma"])
            else:
                prompt_encoded.extend([tokens["end_curly"], tokens["end_curly"]])
                return

        elif param_type == 'string':
            param_str(qwen, prompt_encoded, user_request, tokens)
            if index < len(parameters) - 1:
                prompt_encoded.extend([tokens["comma"]])
            else:
                prompt_encoded.append(tokens['end_curly'])
                prompt_encoded.append(tokens['end_curly'])
                return

        elif param_type == "integer":
            param_int(qwen, prompt_encoded, user_request, tokens)

            if index < len(parameters) - 1:
                prompt_encoded.append(tokens["comma"])
            else:
                prompt_encoded.append(tokens["end_curly"])
                prompt_encoded.append(tokens["end_curly"])

        elif param_type == "number" or param_type == 'float':
            param_float(qwen, prompt_encoded, tokens)

            if index < len(parameters) - 1:
                prompt_encoded.append(tokens["comma"])
            else:
                prompt_encoded.append(tokens["end_curly"])
                prompt_encoded.append(tokens["end_curly"]) 
