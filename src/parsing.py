from typing import Dict, List
from enum import Enum

try:
    from pydantic import BaseModel
except ModuleNotFoundError as e:
    print(f"Module not found: {e.name}")
    exit(1)

class Jsontypes(str, Enum):
    string = "string"
    boolean = "boolean"
    integer = "integer"
    number = "number"

class Parsing(BaseModel):
    functions: List[Dict[str, str | dict]] | Dict[str , str | dict]
    prompts:  dict[str , str] | List[Dict[str, str]] 


    def parse_prompts(self):

        prompts = self.prompts
        if not self.prompts:
            raise ValueError("Empty prompts file !")

        if isinstance(prompts, dict):
            prompts = [prompts]

        for prompt in prompts:
            prompt_keys = list(prompt.keys())

            if len(prompt_keys) != 1:
                raise ValueError(
                    "Respect the Prompts file schema by providing "
                    "only a 'prompt' key for every json object !")

            original = prompt_keys[0]
            key = prompt_keys[0].strip().lower()
            if key != 'prompt':
                raise ValueError("Invalid key !")

            value = prompt[original].strip()
            prompt[original] = value
            if not value:
                raise ValueError("Empty Prompt detected")
            
    @staticmethod
    def validate_unique_function_names(functions):
        seen = set()

        for function in functions:
            name = function['name']
            if name in seen:
                return True
            seen.add(name)

    def parse_functions_schema(self):

        functions = self.functions
        if not functions:
            raise ValueError("Empty functions file !")
        if not isinstance(functions, list):
            functions = [functions]
    
        allowed_keys = {"name", "description", "parameters", "returns"}

        for i, function in enumerate(functions):
            new_function = {}

            for key, value in function.items():
                key = key.strip().lower()

                if key not in allowed_keys:
                    raise ValueError(f"Invalid key '{key}' in function #{i + 1}")

                if isinstance(value, str):
                    value = value.strip()

                new_function[key] = value

            if set(new_function.keys()) != allowed_keys:
                raise ValueError(f"Invalid schema in function #{i + 1}")

            functions[i] = new_function
        self.functions = functions
        return self.functions

    def validate_name(self, name):
        name = name.strip()
        if not name:
            return False
        if not name.isidentifier():
            return False
        return True

    def validate_description(self, description):
        if not description:
            return False
        return True

    def validate_param(self, name, parameter, function):

        new_parameter = {}

        for key, value in parameter.items():
            try:
                new_parameter[key.strip().lower()] = value.strip().lower()
            except Exception:
                return False
        parameter = new_parameter
        # function['parameters'] = parameter


        name = name.strip()

        if not name:
            return False

        if not name.isidentifier():
            return False

        if not isinstance(parameter, dict):
            return False

        if len(parameter) != 1:
            return False

        if "type" not in parameter:
            return False

        try:
            Jsontypes(parameter["type"])
        except ValueError:
            return False

        return True


    def validate_parameters(self, parameters, function):

        if not isinstance(parameters, dict):
            return False

        for name, parameter in parameters.items():
            if not self.validate_param(name, parameter, function):
                return False

        return True

    def validate_functions(self):

        functions = self.functions
        if not isinstance(functions, list):
            functions = [functions]
        
        for i, function in enumerate(functions):
            if not self.validate_name(function["name"]):
                raise ValueError(
                    f"Invalid function name in function #{i + 1}"
                )

            if not self.validate_description(function["description"]):
                raise ValueError(
                    f"Invalid description in function '{function['name']}'"
                )

            if not self.validate_parameters(function["parameters"], function):
                raise ValueError(
                    f"Invalid parameters in function '{function['name']}'"
                )
            try:
                returns = {k.lower(): v.lower().strip() for k, v in function["returns"].items()}
            except Exception:
                raise ValueError("An Error occured in the 'return' area !")

            if not isinstance(returns, dict):
                raise ValueError(
                    f"Invalid returns in function '{function['name']}'"
                )

            if len(returns) != 1:
                raise ValueError(
                    f"Invalid returns schema in function '{function['name']}'"
                )

            if "type" not in returns:
                raise ValueError(
                    f"Missing return type in function '{function['name']}'"
                )

            try:
                returns['type'] = returns['type'].strip()
                Jsontypes(returns["type"])
            except ValueError:
                if returns['type'] == 'none':
                    return True
                raise ValueError(
                    f"Invalid return type in function '{function['name']}'"
                )
            