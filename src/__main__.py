try:
    import argparse
    import json
    from typing import Dict, List, Set, Tuple, Any
    from pydantic import ValidationError
    from pydantic_core import ErrorDetails
    from src.parsing import Parsing
    from llm_sdk import Small_LLM_Model  # type: ignore
except Exception as e:
    print("An error occured during loading the modules !", e)
    exit(1)


def error_on_duplicates(pairs: List[Tuple[str, object]]) -> Dict:
    """Detect duplicate keys while loading a JSON object.
    Args:
        pairs: A List of (key, value) tuples produced by
            ``json.load`` via ``object_pairs_hook``.
    Returns:
        A dictionary containing the parsed key-value pairs.
    Raises:
        ValueError: If a duplicate key is found.
    """
    seen: Set = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"Duplicate key detected: {key}")
        seen.add(key)
    return dict(pairs)


if __name__ == "__main__":
    arg_parser: argparse.ArgumentParser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--functions_definition",
        required=True, type=str,
        help="Enter the path of the functions definitions file")
    arg_parser.add_argument(
        "--input", required=True, help="Enter the path of the prompts file")
    arg_parser.add_argument(
        "--output", required=True, help="Enter the path of the output file")
    arguments: argparse.Namespace = arg_parser.parse_args()
    try:
        with open(arguments.input, 'r') as f:
            prompts: List[Dict] = json.load(
                f, object_pairs_hook=error_on_duplicates)
        with open(arguments.functions_definition, 'r') as f:
            functions: Any = json.load(
                f, object_pairs_hook=error_on_duplicates)
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
            parser: Parsing = Parsing(
                functions=functions,
                prompts=prompts)
            parser.parse_prompts()
            new_functions: Any = parser.parse_functions_schema()
            parser.validate_functions()
        except ValidationError as e:
            err: ErrorDetails = e.errors()[0]
            print(f"Pydantic validation error:\n{err['msg']}")
            exit(1)
        except ValueError as e:
            print(f"Parsing error: {e}")
            exit(1)
        else:
            try:
                qwen: Small_LLM_Model = Small_LLM_Model()
                tokens: Dict[str, (int | List)] = {
                    "quote_points": [int(x) for x in qwen.encode('":')[0]],
                    "comma": [int(x) for x in qwen.encode(', ')[0]][0],
                    "end_curly": [int(x) for x in qwen.encode('}')[0]][0],
                    "quote": [int(x) for x in qwen.encode('"')[0]][0],
                    "string_comma": [int(x) for x in qwen.encode('",')[0]][0],
                    "string_curly": [int(x) for x in qwen.encode('"}')[0]][0]}
            except Exception as e:
                print("Making a Model object failed !", e)
                exit(1)
            else:
                from src.ween import dump_all
                from time import perf_counter
                start: float = perf_counter()
                dump_all(qwen, new_functions, prompts, arguments)
                print("It took :", perf_counter() - start, " seconds !")
