try:
    import argparse, json
    from pydantic import ValidationError
    from parsing import Parsing
    from llm_sdk import Small_LLM_Model
except (ModuleNotFoundError, NameError, ImportError) as e:
    print("An error occured during loading the modules !", e)
    exit(1)
except Exception as e:
    print("An error occured during loading the modules !", e)
    exit(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition", required=True, type=str, help="Enter the path of the functions definitions file")
    parser.add_argument(
        "--input", required=True, help="Enter the path of the prompts file")
    parser.add_argument(
        "--output", required=True, help="Enter the path of the output file")
    arguments = parser.parse_args()
    try:
        with open(arguments.input, 'r') as f:
            prompts = json.load(f)

        with open(arguments.functions_definition, 'r') as f:
            functions = json.load(f)

        with open(arguments.output, 'w') as f:
            pass

    except FileNotFoundError:
        print("The path provided does not lead to an existing file !")
    except PermissionError:
        print("File openning failed due to permissions restrictions !")
    except IsADirectoryError:
        print("The path provided leads to a directory !")
    except json.JSONDecodeError:
        print("The .json file doesn't respect the json structure")
    except Exception:
        print("Error happened !")
    else:

        try:

            parser = Parsing(
            functions=functions,
            prompts=prompts
        )

            parser.parse_prompts()
            parser.parse_functions_schema()
            parser.validate_functions()

        except ValidationError as e:
            err = e.errors()[0]
            print(f"Pydantic validation error:\n{err['msg']}")
            exit(1)

        except ValueError as e:
            print(f"Parsing error: {e}")
            exit(1)
        
        else:
                        
            try:
                qwen = Small_LLM_Model()
                tokens = {
                    "quote_points": qwen.encode('":').tolist()[0],
                    "comma": qwen.encode(', ').tolist()[0][0],
                    "end_curly": qwen.encode('}').tolist()[0][0],
                    "quote": qwen.encode('"').tolist()[0][0],
                    "string_comma": qwen.encode('",').tolist()[0][0],
                    "string_curly": qwen.encode('"}').tolist()[0][0]}
            except Exception:
                print("Making a Model object failed !")
                exit(1)
            else:
                from ween import function_name, prompt_handle, params, get_vocab_str
                user_request = "Replace all vowels in 'Programming is fun' with asterisks"
                user_request = json.dumps(user_request)
                prompt = (
                    "Generate a valid JSON object with exactly three keys: prompt, name (the chosen function), and parameters. "
                    "You must choose the function whose description best matches the user's request. "
                    f"Available functions: {functions} "
                    f"User request: {user_request} "
                    "When a function expects a regex: "
                    "- If the request says \"replace all numbers\", use \"[0-9]+\". "
                    "- If the request says \"replace all vowels\", use \"[AEIOUaeiou]\". "
                    "- If the request says \"replace the word 'X'\", use \"X\" as the regex. "
                    "- The replacement parameter must contain the actual replacement string, not a description. "
                    "\"with NUMBERS\" -> \"NUMBERS\". "
                    "\"with dog\" -> \"dog\". "
                    "\"with asterisks\" -> \"*\". "
                    "Example output: "
                    "{\"prompt\":\"Add 2 and 8\",\"name\":\"fn_add_numbers\",\"parameters\":{\"a\":2,\"b\":8}} "
                    "Answer: {"
                )
                to_cut = len(prompt) - 1
                from time import perf_counter
                start = perf_counter()
                prompt_encoded = qwen.encode(prompt).tolist()[0]
                prompt_handle(qwen, prompt_encoded, user_request)


                the_function = function_name(qwen, prompt_encoded, functions)[:-1]
                the_function = qwen.decode(the_function)
                for f in functions:
                    if f['name'] == the_function:
                        the_function = f
                        break
                
                params(qwen, prompt_encoded, the_function, user_request)
                print(qwen.decode(prompt_encoded)[to_cut:])
                print(perf_counter() - start)
            