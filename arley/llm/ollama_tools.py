import abc
import enum
import sys

from ollama._types import Tool, Message

from arley.Helper import Singleton
from arley.config import settings

import datetime
import json
import uuid
from types import NoneType
from typing import Literal, Any, Union, Type

# from abc import ABC, abstractmethod


from arley import Helper

from loguru import logger

# disable loguru for this module
logger.disable(__name__)  # "arley.llm.ollama_tools")



# heavily inspired by https://github.com/technovangelist/videoprojects/blob/main/2024-07-10-functioncalling-with-tools/tools.ts

# https://www.promptingguide.ai/de/techniques/prompt_chaining
#
# You are a helpful assistant. Your task is to help answer a question given a document. The first step is to extract quotes relevant to the question from the document, delimited by ####. Please output the list of quotes using <quotes></quotes>. Respond with "No relevant quotes found!" if no relevant quotes were found.
# ####
# {{document}}
# ####

# Given a set of relevant quotes (delimited by <quotes></quotes>) extracted from a document and the original document (delimited by ####), please compose an answer to the question. Ensure that the answer is accurate, has a friendly tone, and sounds helpful.
# ####
# {{document}}
# ####
# <quotes>
# - Chain-of-thought (CoT) prompting[27]
# - Generated knowledge prompting[37]
# - Least-to-most prompting[38]
# - Self-consistency decoding[39]
# - Complexity-based prompting[41]
# - Self-refine[42]
# - Tree-of-thought prompting[43]
# - Maieutic prompting[45]
# - Directional-stimulus prompting[46]
# - Textual inversion and embeddings[59]
# - Using gradient descent to search for prompts[61][62][63][64]
# - Prompt injection[65][66][67]
# </quotes>


# TODO HT 20240716 -> adapt as pydantic-type
class OllamaCallableParameter:
    logger = logger.bind(classname=__qualname__)

    def __init__(
        self,
        name: str,
        allowed_value_type: Union[Any, int, float, str, dict, list, NoneType] = Any,
        description: str | None = None,
        required: bool = True,
    ):
        self.name: str = name
        self.allowed_value_type: Union[Any, int, float, str, dict, list, NoneType, enum.Enum] = (
            allowed_value_type  # or permutation of these ?!
        )
        self.required: bool = required
        self.description: str | None = description

    def repr_json(self, ollama_tools_style: bool = False):
        ret: dict
        if not ollama_tools_style:
            ret = {
                "name": self.name,
                "allowed_value_type": (
                    "string" if self.allowed_value_type.__name__ == "str" else self.allowed_value_type.__name__
                ),
                "required": self.required,
                "description": self.description,
            }
        else:
            if isinstance(self.allowed_value_type, enum.Enum):
                enumsubtype: str = "string"
                enumv: enum.Enum = self.allowed_value_type

                if isinstance(self.allowed_value_type, enum.IntEnum):
                    enumsubtype = "int"

                # noinspection PyUnresolvedReferences
                enum_value_list = [e for e in enumv.__members__.values()]

                ret = {
                    self.name: {
                        "type": enumsubtype,
                        "description": self.description,
                        # noinspection PyUnresolvedReferences
                        "enum": enum_value_list,
                    }
                }
            else:
                ret = {
                    self.name: {
                        "type": (
                            "string" if self.allowed_value_type.__name__ == "str" else self.allowed_value_type.__name__
                        ),
                        "description": self.description,
                    }
                }

        return ret


class OllamaCallableParameterValue:
    logger = logger.bind(classname=__qualname__)

    def __init__(self, parameter: OllamaCallableParameter, value: Union[Any, int, float, str, dict, list, NoneType]):
        self.parameter: OllamaCallableParameter = parameter
        self.value: None | str | dict | list = value
        # TODO HT 20240716 type-checking!!!
        # 	-> e.g. adapt as pydantic-type

    def repr_json(self):
        ret: dict = {
            "parameter": self.parameter.repr_json(),
            "value": self.value,
        }

        return ret


class OllamaCallableTool(abc.ABC):
    logger = logger.bind(classname=__qualname__)
    NAME: str
    DESCRIPTION: str
    PARAMETERS: list[OllamaCallableParameter]

    @classmethod
    @abc.abstractmethod
    def execute(cls, values: set[OllamaCallableParameterValue]): ...

    @classmethod
    def repr_json(cls, ollama_tools_style: bool = False) -> dict | Tool:
        ret: dict | Tool

        if not ollama_tools_style:
            ret: dict = {
                "name": cls.NAME,
                "description": cls.DESCRIPTION,
                "parameters": [k.repr_json(ollama_tools_style=False) for k in cls.PARAMETERS],
            }
        else:
            prop_dict: dict = {}
            for k in cls.PARAMETERS:
                prop_dict.update(k.repr_json(ollama_tools_style=True))

                cls.logger.debug(f"PROP_DICT:\n{Helper.get_pretty_dict_json_no_sort(prop_dict)}")

            ret: Tool = {
                "type": "function",
                "function": {
                    "name": cls.NAME,
                    "description": cls.DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "properties": prop_dict,
                        "required": [k.name for k in cls.PARAMETERS if k.required],
                    },
                },
            }

        return ret


class ToolBox(metaclass=Singleton):
    logger = logger.bind(classname=__qualname__)

    def __init__(self):
        if "available_tools" in vars(self):
            logger.debug(f"BEFORE SUPER: {self.available_tools=}")
        super(ToolBox, self).__init__()
        if "available_tools" in vars(self):
            logger.debug(f"AFTER SUPER: {self.available_tools=}")

        if "available_tools" not in vars(self):
            self.available_tools: set[Type[OllamaCallableTool]] = set()

        logger.debug(f"AFTER INIT: {self.available_tools=}")

    @classmethod
    def get_instance(cls):
        return cls()

    def register_tool(self, tool: Type[OllamaCallableTool]):
        self.available_tools.add(tool)

    def un_register_tool(self, tool: Type[OllamaCallableTool]):
        self.available_tools.remove(tool)

    def get_available_tools(self) -> list[Type[OllamaCallableTool]]:
        return [k for k in self.available_tools]

    def execute_tool_function(self, function_name: str, parameters: list[dict[str, str]] | dict[str, Any]):
        for tool in self.get_available_tools():
            if tool.NAME == function_name:
                provided_params: dict[str, Union[Any, int, float, str, dict, list, NoneType, enum.EnumType]] = {}

                if isinstance(parameters, dict):
                    for param_name, param_value in parameters.items():
                        provided_params[param_name] = param_value
                else:
                    for param in parameters:
                        provided_params[param["parameterName"]] = param["parameterValue"]

                applied_params: set[OllamaCallableParameterValue] = set()
                for defined_parameter in tool.PARAMETERS:
                    if defined_parameter.name in provided_params:
                        applied_param: OllamaCallableParameterValue = OllamaCallableParameterValue(
                            parameter=defined_parameter, value=provided_params[defined_parameter.name]
                        )
                        # applied_params[defined_parameter.name] = applied_param
                        applied_params.add(applied_param)
                    elif defined_parameter.required:
                        raise RuntimeError(f"not supplied parameter {defined_parameter.name=} for {function_name=}")

                # EXECUTE!
                execute_result = tool.execute(values=applied_params)

                return execute_result


class CityToLatLonTool(OllamaCallableTool):
    logger = logger.bind(classname=__qualname__)
    NAME: str = "CityToLatLon"
    DESCRIPTION: str = "Get the latitude and longitude for a given city"
    PARAMETERS: list[OllamaCallableParameter] = [
        OllamaCallableParameter(name="city", allowed_value_type=str, description="Name of the city", required=True)
    ]

    @classmethod
    def execute(cls, values: set[OllamaCallableParameterValue]):
        cls.logger.debug(f"{cls.NAME}::execute:")
        for va in values:
            cls.logger.debug(f"{cls.NAME}::execute: {Helper.get_pretty_dict_json_no_sort(va.repr_json())}")
        # async function CityToLatLon(city: string) {
        # 	const output = await fetch(
        # 		`https://nominatim.openstreetmap.org/search?q=${city}&format=json`,
        # 	);
        # 	const json = await output.json();
        # 	return [json[0].lat, json[0].lon];
        # }


class Now(OllamaCallableTool):
    logger = logger.bind(classname=__qualname__)
    NAME: str = "GetTimeAndDateOfNow"
    DESCRIPTION: str = "Get the current time and/or get the current date"
    PARAMETERS: list[OllamaCallableParameter] = []

    @classmethod
    def execute(cls, values: set[OllamaCallableParameterValue]):
        cls.logger.debug(f"{cls.NAME}::execute:")
        for va in values:
            cls.logger.debug(f"{cls.NAME}::execute: {Helper.get_pretty_dict_json_no_sort(va.repr_json())}")
        # async function CityToLatLon(city: string) {
        # 	const output = await fetch(
        # 		`https://nominatim.openstreetmap.org/search?q=${city}&format=json`,
        # 	);
        # 	const json = await output.json();
        # 	return [json[0].lat, json[0].lon];
        # }


class WeatherFromLatLonTool(OllamaCallableTool):
    logger = logger.bind(classname=__qualname__)
    NAME: str = "WeatherFromLatLon"
    DESCRIPTION: str = "Get the weather for a location"
    PARAMETERS: list[OllamaCallableParameter] = [
        OllamaCallableParameter(
            name="latitude", allowed_value_type=float, description="The latitude of the location", required=True
        ),
        OllamaCallableParameter(
            name="longitude", allowed_value_type=float, description="The longitude of the location", required=True
        ),
    ]

    @classmethod
    def execute(cls, values: set[OllamaCallableParameterValue]):
        cls.logger.debug(f"{cls.NAME}::execute:")
        for va in values:
            cls.logger.debug(f"{cls.NAME}::execute: {Helper.get_pretty_dict_json_no_sort(va.repr_json())}")
        # async function WeatherFromLatLon(latitude: string, longitude: string) {
        # 	const output = await fetch(
        # 		`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m&temperature_unit=fahrenheit&wind_speed_unit=mph&forecast_days=1`,
        # 	);
        #
        # 	const json = await output.json();
        #   console.log(`${json.current.temperature_2m} degrees Farenheit`);
        # }


class LatlonToCityTool(OllamaCallableTool):
    logger = logger.bind(classname=__qualname__)
    NAME: str = "LatLonToCity"
    DESCRIPTION: str = "Get the city name for a given latitude and longitude"
    PARAMETERS: list[OllamaCallableParameter] = [
        OllamaCallableParameter(
            name="latitude", allowed_value_type=float, description="The latitude of the location", required=True
        ),
        OllamaCallableParameter(
            name="longitude", allowed_value_type=float, description="The longitude of the location", required=True
        ),
    ]

    @classmethod
    def execute(cls, values: set[OllamaCallableParameterValue]):
        cls.logger.debug(f"{cls.NAME}::execute:")
        for va in values:
            cls.logger.debug(f"\t{cls.NAME}::execute: {Helper.get_pretty_dict_json_no_sort(va.repr_json())}")
        # async function LatLonToCity(latitude: string, longitude: string) {
        # 	const output = await fetch(
        # 		`https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json`,
        # 	);
        # 	const json = await output.json();
        # 	console.log(json.display_name);
        # }


class WebSearchTool(OllamaCallableTool):
    logger = logger.bind(classname=__qualname__)
    NAME: str = "WebSearch"
    DESCRIPTION: str = "Search the web for a query"
    PARAMETERS: list[OllamaCallableParameter] = [
        OllamaCallableParameter(
            name="query", allowed_value_type=str, description="The query to search for", required=True
        )
    ]

    @classmethod
    def execute(cls, values: set[OllamaCallableParameterValue]):
        cls.logger.debug(f"{cls.NAME}::execute:")
        for va in values:
            cls.logger.debug(f"{cls.NAME}::execute: {Helper.get_pretty_dict_json_no_sort(va.repr_json())}")
        # async function WebSearch(query: string) {
        # 	const output = await fetch(
        # 		`http://localhost:3333/search?q=${query}&format=json`,
        # 	);
        # 	const json = await output.json();
        # 	console.log(`${json.results[0].title}\n${json.results[0].content}\n`);
        # }


class WeatherFromLocationTool(OllamaCallableTool):
    logger = logger.bind(classname=__qualname__)
    NAME: str = "WeatherFromLocation"
    DESCRIPTION: str = "Get the weather for a location"
    PARAMETERS: list[OllamaCallableParameter] = [
        OllamaCallableParameter(
            name="location", allowed_value_type=str, description="The location to get the weather for", required=True
        )
    ]

    @classmethod
    def execute(cls, values: set[OllamaCallableParameterValue]):
        cls.logger.debug(f"{cls.NAME}::execute:")
        for va in values:
            cls.logger.debug(f"{cls.NAME}::execute: {Helper.get_pretty_dict_json_no_sort(va.repr_json())}")
        # async function WeatherFromLocation(location: string) {
        #   const latlon = await CityToLatLon(location);
        #   await WeatherFromLatLon(latlon[0], latlon[1]);
        # }


# TOOLS: list[Type[OllamaCallableTool]] = [
# 	Now,
# 	WeatherFromLocationTool,
# 	WeatherFromLatLonTool,
# 	WebSearchTool,
# 	LatlonToCityTool,
# 	CityToLatLonTool
# ]

ToolBox.get_instance().register_tool(Now)
ToolBox.get_instance().register_tool(WeatherFromLocationTool)
ToolBox.get_instance().register_tool(WeatherFromLatLonTool)
ToolBox.get_instance().register_tool(WebSearchTool)
ToolBox.get_instance().register_tool(LatlonToCityTool)
ToolBox.get_instance().register_tool(CityToLatLonTool)


TOOLSSTRING: str = (
    "```json\n"
    + Helper.get_pretty_dict_json_no_sort(
        {"tools": [k.repr_json(ollama_tools_style=False) for k in ToolBox.get_instance().get_available_tools()]}
    )
    + "\n```"
)

TOOLSLIST_OLLAMA_TOOL_STYLE: list[Tool] = [
    k.repr_json(ollama_tools_style=True) for k in ToolBox.get_instance().get_available_tools()
]


FUNCTION_SCHEMA: dict = {
    "functionName": "function name",
    "parameters": [
        {
            "parameterName": "name of parameter",
            "allowed_value_type": "string or float or int",
            "parameterValue": "value of parameter - do not use apostrophes here if allowed_value_type is not string",
        }
    ],
}


# 	switch (functionName) {
# 		case "WeatherFromLocation":
# 			return await WeatherFromLocation(getValueOfParameter("location", parameters));
# 		case "WeatherFromLatLon":
# 			return await WeatherFromLatLon(
# 				getValueOfParameter("latitude", parameters),
# 				getValueOfParameter("longitude", parameters),
# 			);
# 		case "WebSearch":
# 			return await WebSearch(getValueOfParameter("query", parameters));
# 		case "LatLonToCity":
# 			return await LatLonToCity(
# 				getValueOfParameter("latitude", parameters),
# 				getValueOfParameter("longitude", parameters),
# 			);
# 	}
# }

# function getValueOfParameter(
# 	parameterName: string,
# 	parameters: FunctionParameter[],
# ) {
# 	return parameters.filter((p) => p.parameterName === parameterName)[0]
# 		.parameterValue;
# }
#


class GetUrl(OllamaCallableTool):
    ...


def _ollama_test_llama31_with_tools():
    from arley.llm.ollama_adapter import _get_fc_call_generate_priming_history

    # https://github.com/ollama/ollama-python/blob/main/examples/tools/main.py
    # TODO HT 20250324 update to conform to: https://github.com/ollama/ollama-python/blob/main/examples/tools.py

    from arley.llm.ollama_adapter import ask_ollama_chat

    fc_questions: dict = {}
    # just recycling the priming questions...
    msgs: list[Message] = _get_fc_call_generate_priming_history(
        function_schema=FUNCTION_SCHEMA,
        toolstring=TOOLSSTRING,
        allow_backticked_json=False
    )
    for i in range(1, len(msgs), 2):
        msg: Message = msgs[i]
        msgn: Message = msgs[i + 1]

        fc_questions[msg["content"]] = msgn["content"]

    for fctest_question in fc_questions:
        logger.debug(f"Q: {fctest_question}")

        resp2: dict = ask_ollama_chat(
            return_format="",
            system_prompt=None,
            prompt=fctest_question,
            msg_history=None,
            model="llama3.1:latest", # "llama3.1:70b-instruct-q4_0",  # "llama3.1:latest", "llama3.1:70b-instruct-q3_K_M"
            evict=False,
            temperature=0.0,
            # penalize_newline=True,
            # repeat_penalty=1.0,
            top_k=40,
            top_p=0.9,
            num_predict=-1,
            seed=0,
            print_msgs=False,
            print_response=True,
            print_options=False,
            tools=TOOLSLIST_OLLAMA_TOOL_STYLE,
            print_chunks_when_streamed=False,
            streamed_print_to_io=sys.stdout,
            streamed=False
        )

        if "tool_calls" in resp2["message"]:
            tc: Message.ToolCall
            for tc in resp2["message"]["tool_calls"]:
                function: Message.ToolCall.Function = tc["function"]  # was: ToolCallFunction
                # ingore
                ToolBox.get_instance().execute_tool_function(function["name"], function["arguments"])

# function calling in hermes3: https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B

# <|im_start|>system
# You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> {"type": "function", "function": {"name": "get_stock_fundamentals", "description": "get_stock_fundamentals(symbol: str) -> dict - Get fundamental data for a given stock symbol using yfinance API.\\n\\n    Args:\\n        symbol (str): The stock symbol.\\n\\n    Returns:\\n        dict: A dictionary containing fundamental data.\\n            Keys:\\n                - \'symbol\': The stock symbol.\\n                - \'company_name\': The long name of the company.\\n                - \'sector\': The sector to which the company belongs.\\n                - \'industry\': The industry to which the company belongs.\\n                - \'market_cap\': The market capitalization of the company.\\n                - \'pe_ratio\': The forward price-to-earnings ratio.\\n                - \'pb_ratio\': The price-to-book ratio.\\n                - \'dividend_yield\': The dividend yield.\\n                - \'eps\': The trailing earnings per share.\\n                - \'beta\': The beta value of the stock.\\n                - \'52_week_high\': The 52-week high price of the stock.\\n                - \'52_week_low\': The 52-week low price of the stock.", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}}  </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
# <tool_call>
# {"arguments": <args-dict>, "name": <function-name>}
# </tool_call><|im_end|>

# https://github.com/NousResearch/Hermes-Function-Calling

if __name__ == "__main__":
    # logger.debug(TOOLSSTRING)
    logger.debug(Helper.get_pretty_dict_json_no_sort(TOOLSLIST_OLLAMA_TOOL_STYLE))
    _ollama_test_llama31_with_tools()
