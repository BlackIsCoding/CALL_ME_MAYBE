import argparse, json

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
    # else:
    #     try:
    #         prompt = "" \
    #         "Qwen , you are a function calling assistant, i will send you mutliple " \
    #         "functions definitions and you will try to find wich function to use and return it's name.\n"\
    #         f"Available functions: {functions}\n"\
    #         f"User request: {prompts[0]}\n"\
    #         "Answer: "
    #     except KeyboardInterrupt:
    #         print("You Killed the Process")
    #     except Exception:
    #         print("Error detected")