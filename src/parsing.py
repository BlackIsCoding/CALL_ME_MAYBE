from load import functions, prompts
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
    functions: List[Dict[str, str | dict]]
    prompts: List[Dict[str, str]]


    def parse_prompts(self):

        prompts = self.prompts
        if not self.prompts:
            raise ValueError("Empty prompts file !")

        for prompt in prompts:
            prompt_keys = list(prompt.keys())

            if len(prompt_keys) != 1:
                raise ValueError(
                    "Respect the Prompts file schema by providing "
                    "only a 'prompt' key for every json object !")

            key = prompt_keys[0].strip().lower()
            if key != 'prompt':
                raise ValueError("Invalid key !")

            value = prompt['prompt'].strip()
            prompt['prompt'] = value
            if not value:
                raise ValueError("Empty Prompt detected")

    def parse_functions_schema(self):

        functions = self.functions
        if not functions:
            raise ValueError("Empty functions file !")
    
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

    def validate_name(self, name):
        if not name:
            return False
        if not name.isidentifier():
            return False
        return True

    def validate_description(self, description):
        if not description:
            return False
        return True

    def validate_param(self, name, parameter):
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


    def validate_parameters(self, parameters):
        if not parameters:
            return False

        if not isinstance(parameters, dict):
            return False

        for name, parameter in parameters.items():
            if not self.validate_param(name, parameter):
                return False

        return True
        