try:
    import argparse, json
    from pydantic import ValidationError
    from .parsing import Parsing
    from llm_sdk import Small_LLM_Model
except (ModuleNotFoundError, NameError, ImportError) as e:
    print("An error occured during loading the modules !", e)
    exit(1)
except Exception as e:
    print("An error occured during loading the modules !", e)
    exit(1)

def error_on_duplicates(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"Duplicate key detected: {key}")
        seen.add(key)
    return dict(pairs)

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
            prompts = json.load(f, object_pairs_hook=error_on_duplicates)

        with open(arguments.functions_definition, 'r') as f:
            functions = json.load(f, object_pairs_hook=error_on_duplicates)

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
    except ValueError:
        print("Error happened due to duplicate keys")
    except Exception:
        print("Error happened !")
    else:

        try:

            parser = Parsing(
            functions=functions,
            prompts=prompts
        )

            parser.parse_prompts()
            functions = parser.parse_functions_schema()
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
                from .ween import dump_all
                from time import perf_counter
                start = perf_counter()
                dump_all(qwen, functions, prompts, arguments)
                print("It took :", perf_counter() - start)
            