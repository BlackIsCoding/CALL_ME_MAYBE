from typing import Any, cast, List
from enum import Enum
from pydantic import BaseModel


class Jsontypes(str, Enum):

    """Supported JSON data types for schema validation."""
    string = "string"
    boolean = "boolean"
    integer = "integer"
    number = "number"


class Parsing(BaseModel):
    """Validate and parse function definitions and user prompts."""

    functions: (
        List[dict[str, str | dict[str, Any]]]
        | dict[str, str | dict[str, Any]]
    )
    prompts: dict[str, str] | List[dict[str, str]]

    def parse_prompts(self) -> None:
        """Validate and normalize the prompts structure.

        Raises:
            ValueError: If the prompts collection is empty, if a prompt
                object does not have exactly one key, if that key is not
                ``'prompt'``, or if the prompt value is empty.
        """

        prompts: dict[str, str] | List[dict[str, str]] = self.prompts

        if not self.prompts:
            raise ValueError("Empty prompts file !")

        prompt_List: List[dict[str, str]]

        if isinstance(prompts, dict):
            prompt_List = [prompts]
        else:
            prompt_List = prompts

        for prompt in prompt_List:
            prompt_keys: List[str] = list(prompt.keys())

            if len(prompt_keys) != 1:
                raise ValueError(
                    "Respect the Prompts file schema by providing "
                    "only a 'prompt' key for every json object !"
                )

            original: str = prompt_keys[0]
            key: str = prompt_keys[0].strip().lower()

            if key != 'prompt':
                raise ValueError("Invalid key !")

            value: str = prompt[original].strip()
            prompt[original] = value

            if not value:
                raise ValueError("Empty Prompt detected")

    @staticmethod
    def validate_unique_function_names(
        functions: List[dict[str, Any]]
    ) -> bool:
        """Check whether duplicated function names exist.

        Args:
            functions: The List of function definitions to inspect.

        Returns:
            True if a duplicate function name is found, False otherwise.
        """

        seen: set[str] = set()

        for function in functions:
            name: str = function['name']

            if name in seen:
                return True

            seen.add(name)

        return False

    def parse_functions_schema(self) -> List[dict[str, str | dict[str, Any]]]:
        """Normalize and validate function schema keys.

        Returns:
            The normalized List of function definitions.

        Raises:
            ValueError: If the functions collection is empty, if a
                function contains a key outside the allowed schema keys,
                or if a function is missing one of the allowed schema
                keys.
        """

        functions: (
            List[dict[str, str | dict[str, Any]]]
            | dict[str, str | dict[str, Any]]
        ) = self.functions

        if not functions:
            raise ValueError("Empty functions file !")

        function_List: List[dict[str, str | dict[str, Any]]]

        # Correct runtime type checks: compare against builtin types
        if isinstance(functions, dict):
            function_List = [functions]
        elif isinstance(functions, list):
            function_List = functions
        else:
            raise TypeError("Invalid type for 'functions' field")

        allowed_keys: set[str] = {
            "name",
            "description",
            "parameters",
            "returns"
        }

        normalized_functions: List[dict[str, str | dict[str, Any]]] = []

        for i, function in enumerate(function_List):

            new_function: dict[str, Any] = {}

            for raw_key, raw_value in function.items():

                key = raw_key.strip().lower()

                if key not in allowed_keys:
                    raise ValueError(
                        f"Invalid key '{key}' in function #{i + 1}"
                    )

                value: str | dict[str, Any] = raw_value

                if isinstance(value, str):
                    value = value.strip()

                new_function[key] = value

            if set(new_function.keys()) != allowed_keys:
                raise ValueError(
                    f"Invalid schema in function #{i + 1}"
                )

            normalized_functions.append(new_function)

        self.functions = cast(
            List[dict[str, str | dict[str, Any]]], normalized_functions
        )

        return normalized_functions

    def validate_name(self, name: str) -> bool:
        """Validate a function name format.

        Args:
            name: The function name to validate.

        Returns:
            True if the name is a non-empty valid Python identifier,
            False otherwise.
        """

        name = name.strip()

        if not name:
            return False

        if not name.isidentifier():
            return False

        return True

    def validate_description(self, description: str) -> bool:
        """Validate a function description.

        Args:
            description: The function description to validate.

        Returns:
            True if the description is non-empty, False otherwise.
        """

        if not description:
            return False

        return True

    def validate_param(
        self,
        name: str,
        parameter: dict[str, Any]
    ) -> bool:
        """Validate a function parameter schema.

        Args:
            name: The parameter name to validate.
            parameter: The parameter schema, expected to contain a
                single ``'type'`` key with a supported JSON type value.

        Returns:
            True if the parameter name and schema are valid, False
            otherwise.
        """

        new_parameter: dict[str, str] = {}

        for key, value in parameter.items():

            try:
                new_parameter[key.strip().lower()] = value.strip().lower()

            except Exception:
                return False

        normalized_parameter: dict[str, str] = new_parameter

        name = name.strip()

        if not name:
            return False

        if not name.isidentifier():
            return False

        if not isinstance(normalized_parameter, dict):
            return False

        if len(normalized_parameter) != 1:
            return False

        if "type" not in normalized_parameter:
            return False

        try:
            Jsontypes(normalized_parameter["type"])

        except ValueError:
            return False

        return True

    def validate_parameters(
        self,
        parameters: dict[str, Any]
    ) -> bool:
        """Validate all parameters of a function.

        Args:
            parameters: A mapping of parameter names to their schema.

        Returns:
            True if every parameter is valid, False otherwise.
        """

        if not isinstance(parameters, dict):
            return False

        for name, parameter in parameters.items():

            if not self.validate_param(name, parameter):
                return False

        return True

    def validate_functions(self) -> bool:
        """Validate all function definitions.

        Returns:
            True or False depending on the validated return type of the
            first processed function (see raised errors for invalid
            schemas).

        Raises:
            ValueError: If a function has an invalid name, description,
                parameters, or returns schema.
        """

        functions: (
            List[dict[str, str | dict[str, Any]]]
            | dict[str, str | dict[str, Any]]
        ) = self.functions

        function_List: List[dict[str, str | dict[str, Any]]]

        # Use builtin types for runtime checks
        if isinstance(functions, dict):
            function_List = [functions]
        elif isinstance(functions, list):
            function_List = functions
        else:
            raise TypeError("Invalid type for 'functions' field")

        for i, function in enumerate(function_List):

            name = cast(str, function["name"])
            description = cast(str, function["description"])
            parameters = cast(dict[str, Any], function["parameters"])

            if not self.validate_name(name):
                raise ValueError(
                    f"Invalid function name in function #{i + 1}"
                )

            if not self.validate_description(description):
                raise ValueError(
                    f"Invalid description in function '{name}'"
                )

            if not self.validate_parameters(parameters):
                raise ValueError(
                    f"Invalid parameters in function '{name}'"
                )

            try:
                returns: dict[str, str] = {
                    k.lower(): v.lower().strip()
                    for k, v in cast(
                        dict[str, str], function["returns"]
                    ).items()
                }

            except Exception:
                raise ValueError(
                    "An Error occured in the 'return' area !"
                )

            if not isinstance(returns, dict):
                raise ValueError(
                    f"Invalid returns in function '{name}'"
                )

            if len(returns) != 1:
                raise ValueError(
                    f"Invalid returns schema in function '{name}'"
                )

            if "type" not in returns:
                raise ValueError(
                    f"Missing return type in function '{name}'"
                )

            try:
                returns['type'] = returns['type'].strip()
                Jsontypes(returns["type"])

            except ValueError:

                if returns['type'] == 'none':
                    return True

                raise ValueError(
                    f"Invalid return type in function '{name}'"
                )

            return False

        return False
