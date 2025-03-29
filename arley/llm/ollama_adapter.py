import textwrap
from functools import partial
from pathlib import Path
from typing import Union

# if TYPE_CHECKING:
#     from _typeshed import SupportsWrite

from io import StringIO

from jinja2 import Environment, FileSystemLoader
# from pprint import pprint

from ollama import Message
from ollama._types import Tool

from arley import Helper


from arley.config import (
    TemplateType,
    OLLAMA_HOST,
    get_ollama_options,
    TEMPLATEDIRPATH
)

import json
import sys
import threading
import time
from typing import Sequence, Optional, Literal, Mapping, Any, Iterator

import ollama

from loguru import logger

from arley.llm.ollama_tools import TOOLSSTRING, FUNCTION_SCHEMA


logger.debug(f"OLLAMA_HOST: {OLLAMA_HOST}")
OLLAMA_CLIENT = ollama.Client(host=OLLAMA_HOST)



# def ask_ollama_generate(
#         prompt: str,
#         system_prompt: str = '',
#         streamed: bool = False,
#         model: str = "llama3:instruct",
#         evict: bool = False,
#         temperature: float = 0.8,
#         top_k: int = 40,
#         top_p: float = 0.9,
#         num_predict: int = 128,
#         seed: int = 0,
#         return_format: Literal["json", ""] = "",
#         print_response: bool = False,
#         context_in: Optional[Sequence[int]] = None,
#         streamed_print_to_io: "SupportsWrite[str] | None" = sys.stdout,
#         print_options: bool = False,
#         keep_alive: int = 300,  # was: -1
#         max_tries_ollama_done_response: int = 21,
#         tools: Optional[Sequence[Tool]] = None,
#         print_chunks_when_streamed: bool = False
# ) -> dict | Mapping[str, Any] | Iterator[Mapping[str, Any]]:
#     options: dict = get_ollama_options(model=model,
#                                        top_k=top_k,
#                                        top_p=top_p,
#                                        seed=seed,
#                                        temperature=temperature,
#                                        num_predict=num_predict
#                                        )
#
#     if print_options:
#         logger.debug(Helper.get_pretty_dict_json_no_sort(options))
#
#     # options = None
#     keep_alive: float = keep_alive  # default: 300  # 0: direkt wieder aus dem gpu löschen | -1: keep forever
#     if evict:
#         keep_alive = 0
#
#     ret: Optional[dict[str, Any]] = None
#     output: Optional[Union[Mapping[str, Any], Iterator[Mapping[str, Any]]]] = None
#     for i in range(1, max_tries_ollama_done_response):
#         output = OLLAMA_CLIENT.generate(  # type: ignore
#             model=model,
#             # system=system_prompt,
#             prompt=prompt,
#             context=context_in,
#             stream=streamed,
#             format=return_format,
#             options=options,
#             keep_alive=keep_alive
#         )
#
#         if isinstance(output, Mapping):
#             if output["done"]:
#                 output["loopcount"] = i  # type: ignore
#
#                 tokens_per_second = output["eval_count"] / (output["eval_duration"] / 1000000000)
#                 load_dur_seconds = output["load_duration"] / 1000000000
#                 total_dur_seconds = output["total_duration"] / 1000000000
#
#                 ret = {}
#
#                 ret.update(output)
#
#                 ret["load_dur_seconds"] = load_dur_seconds
#                 ret["total_dur_seconds"] = total_dur_seconds
#                 ret["tokens_per_second"] = tokens_per_second
#
#                 break
#             else:
#                 logger.debug(Helper.get_pretty_dict_json_no_sort(output))
#                 logger.debug(f"LOOP#{i} NO DONE - element received!!!")
#                 time.sleep(5)
#
#     # if streamed...
#     if isinstance(output, Iterator):
#         # streamed_print_to_io = sys.stdout
#         out: StringIO = StringIO()
#         # TODO HT 20240726 -> add another maximum loopcount here ?!
#         chunkidx: int
#         chunk: Mapping[str, Any]
#
#         for chunkidx, chunk in enumerate(output):
#             if print_chunks_when_streamed:
#                 logger.debug(f"STREAMED_CHUNK#{chunkidx + 1}:\n{Helper.get_pretty_dict_json_no_sort(chunk)}")
#
#             chunkdata: str = chunk["response"]
#             out.write(chunkdata)
#             if streamed_print_to_io:
#                 print_me(chunkdata, end="", flush=True, file=streamed_print_to_io)
#                 # time.sleep(1)
#
#             if chunk["done"] == True:
#                 break
#
#         logger.debug(Helper.get_pretty_dict_json_no_sort(chunk))
#
#         tokens_per_second = chunk["eval_count"] / (chunk["eval_duration"] / 1000000000)
#         load_dur_seconds = chunk["load_duration"] / 1000000000
#         total_dur_seconds = chunk["total_duration"] / 1000000000
#
#         if streamed_print_to_io:
#             print_me(flush=True, file=streamed_print_to_io)
#
#         if chunk is not None:
#             ret = {}
#
#             ret.update(chunk)
#             ret["response"] = out.getvalue()
#             ret["chunk_count"] = chunkidx + 1
#
#             ret["load_dur_seconds"] = load_dur_seconds
#             ret["total_dur_seconds"] = total_dur_seconds
#             ret["tokens_per_second"] = tokens_per_second
#
#
#     if print_response:
#         logger.debug(Helper.get_pretty_dict_json_no_sort(ret))
#
#     return ret

def ask_ollama_chat(
    system_prompt: str | None,
    prompt: str,
    # initial_topic: str | None,
    # lang: Literal["de", "en"] | None,
    streamed: bool = False,
    msg_history: Optional[Sequence[Message]] = None,
    model: str = "llama3:instruct",
    evict: bool = False,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    num_predict: int = 128,
    repeat_penalty: float = 1.1,
    # penalize_newline: bool = True,
    seed: int = 0,
    return_format: Literal["json", ""] = "",
    print_msgs: bool = False,
    print_response: bool = False,
    # context_in: Optional[Sequence[int]] = None,
    streamed_print_to_io: "SupportsWrite[str] | None" = sys.stdout,
    print_options: bool = False,
    keep_alive: int = 300,  # was: -1
    max_tries_ollama_done_response: int = 21,
    # tools: ToolBox | None = None,
    tools: Optional[Sequence[Tool]] = None,
    print_chunks_when_streamed: bool = False
) -> dict | Mapping[str, Any] | Iterator[Mapping[str, Any]]:
    """
    Function to request a response from ollama.

    Args:
        system_prompt (str | None): System prompt for the model. Default is None.
        prompt (str): Prompt for the model.
        streamed (bool): Whether to stream the output. Default is False.
        msg_history (Optional[Sequence[Message]]): Message history for the model. Default is None.
        model (str): Model name for the model. Default is "llama3:instruct".
        evict (bool): Whether to evict the model. Default is False.
        temperature (float): Temperature for the model. Default is 0.8.
        top_k (int): Top k for the model. Default is 40.
        top_p (float): Top p for the model. Default is 0.9.
        num_predict (int): Number of predictions for the model. Default is 128.
        seed (int): Seed for the model. Default is 0.
        return_format (Literal["json", ""]): Format for the output. Default is "".
        print_msgs (bool): Whether to print messages. Default is False.
        print_response (bool): Whether to print response. Default is False.
        streamed_print_to_io (SupportsWrite[str] | None, optional): An IO object or file-like object where the text should be printed. Defaults to sys.stdout.
        print_options: lslsl
        keep_alive:  lalala
        max_tries_ollama_done_response: lala
        tools: lala

    Returns:
        dict: Dictionary containing the output of the function.
        :param print_chunks_when_streamed:
    """

    options: dict = get_ollama_options(model=model,
                                       top_k=top_k,
                                       top_p=top_p,
                                       seed=seed,
                                       temperature=temperature,
                                       num_predict=num_predict,
                                       repeat_penalty=repeat_penalty
                                       )

    if print_options:
        logger.debug(Helper.get_pretty_dict_json_no_sort(options))

    # options = None
    keep_alive: float = keep_alive  # default: 300  # 0: direkt wieder aus dem gpu löschen | -1: keep forever
    if evict:
        keep_alive = 0

    msgs: list[Message] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})

    if msg_history:
        for it in msg_history:
            msgs.append(it)

    msgs.append({"role": "user", "content": prompt})

    # if not msg_history and not system_prompt:
    #     raise RuntimeError("Neither msg_history nor system_prompt is given...")

    if print_msgs:
        for msg in msgs:
            logger.debug(f"ROLE: {msg["role"]}")
            logger.debug(f"CONTENT:\n{textwrap.indent(msg["content"], "\t")}")

        #logger.debug(Helper.get_pretty_dict_json_no_sort(msgs))
        # Helper.print_pretty_dict_json(msgs)


    ret: Optional[dict[str, Any]] = None
    output: Optional[Union[Mapping[str, Any], Iterator[Mapping[str, Any]]]] = None
    for i in range(1, max_tries_ollama_done_response):
        output = OLLAMA_CLIENT.chat(  # type: ignore
            model=model,
            messages=msgs,
            tools=tools,
            stream=streamed,
            format=return_format,
            options=options,
            keep_alive=keep_alive
        )

        if isinstance(output, Mapping):
            if output["done"]:
                output["loopcount"] = i  # type: ignore

                tokens_per_second = output["eval_count"] / (output["eval_duration"] / 1000000000)
                load_dur_seconds = output["load_duration"] / 1000000000
                total_dur_seconds = output["total_duration"] / 1000000000

                ret = {}

                ret.update(output)

                ret["load_dur_seconds"] = load_dur_seconds
                ret["total_dur_seconds"] = total_dur_seconds
                ret["tokens_per_second"] = tokens_per_second

                break
            else:
                logger.debug(Helper.get_pretty_dict_json_no_sort(output))
                logger.debug(f"LOOP#{i} NO DONE - element received!!!")
                time.sleep(5)

    # if streamed...
    if isinstance(output, Iterator):
        # streamed_print_to_io = sys.stdout
        out: StringIO = StringIO()
        # TODO HT 20240726 -> add another maximum loopcount here ?!
        for chunkidx, chunk in enumerate(output):
            if print_chunks_when_streamed:
                logger.debug(f"STREAMED_CHUNK#{chunkidx+1}:\n{Helper.get_pretty_dict_json_no_sort(chunk)}")

            chunkdata: str = chunk["message"]["content"]
            out.write(chunkdata)
            if streamed_print_to_io:
                print_me(chunkdata, end="", flush=True, file=streamed_print_to_io)
                # time.sleep(1)

            if chunk["done"] == True:
                tokens_per_second = chunk["eval_count"] / (chunk["eval_duration"] / 1000000000)
                load_dur_seconds = chunk["load_duration"] / 1000000000
                total_dur_seconds = chunk["total_duration"] / 1000000000

                if streamed_print_to_io:
                    print_me(flush=True, file=streamed_print_to_io)

                ret = {}

                ret.update(chunk)
                ret["message"]["content"] = out.getvalue()
                ret["chunk_count"] = chunkidx+1

                ret["load_dur_seconds"] = load_dur_seconds
                ret["total_dur_seconds"] = total_dur_seconds
                ret["tokens_per_second"] = tokens_per_second

                # WANT: response["message"]["content"]
                break


    if ret is None and chunk is not None and out is not None:
        ret = {}

        ret.update(chunk)
        ret["message"]["content"] = out.getvalue()
        ret["chunk_count"] = chunkidx + 1
        ret["CHUNK_RESP_RESPONSE"] = True


    if print_response:
        logger.debug(Helper.get_pretty_dict_json_no_sort(ret))

    return ret


def get_available_models():
    # curl http://localhost:11434/api/tags
    # "models": [
    #     {
    #       "name": "codellama:13b",
    #       "modified_at": "2023-11-04T14:56:49.277302595-07:00",
    #       "size": 7365960935,
    #       "digest": "9f438cb9cd581fc025612d27f7c1a6669ff83a8bb0ed86c94fcf4c5440555697",
    #       "details": {
    #         "format": "gguf",
    #         "family": "llama",
    #         "families": null,
    #         "parameter_size": "13B",
    #         "quantization_level": "Q4_0"
    #       }
    #     },
    for k in OLLAMA_CLIENT.list()["models"]:
        logger.info(json.dumps(k))


def _get_fc_call_generate_priming_history(function_schema: dict, toolstring: str, allow_backticked_json: bool = False) -> list[Message]:
    msgs: list[Message] = []

    system_pr: StringIO = StringIO()

    system_pr.write(f"I am a helpful assistant that takes a question and finds the most appropriate tool or tools to execute, "
                       f"along with the parameters required to run the tool. For the tools, I will only choose \"WebSearch\" as a last resort if no other tool can be used to fullfill the request.\n"
                       f"I will make sure to use the proper type of value (e.g. string, float, int) and have in mind, that only string values need to be enclosed with apostrophes.\n"
                       f"Even if I am asked the same question again (or again and again), I will answer as if I would have been asked the question for the first time.\n"
                       f"I will not respond with implementation hints or stubs and absolutely make sure to not invent any tools not given below and to stick rigidly to the functionName and signature of any supplied tool.\n"
                    )

    if allow_backticked_json:
        system_pr.write(
            f"I will make sure to use backticks for 'json' markup when returning json in my response!\n"
        )
    else:
        system_pr.write(f"I will make sure NOT to use backticks or 'json' markup in my responses!\n")
        system_pr.write(f"Also, I will absolutely not put anything else in my response (e.g. \"Note\", \"explanatory text\" etc.) aside from my response within that JSON schema.\n")

    system_pr.write(
        f"I will output as JSON following the following schema rigidly.\n"
        f"\nJSON schema for my responses:\n"
        f"```json\n{Helper.get_pretty_dict_json_no_sort(function_schema)}\n```\n"
        f"\nThe tools are:\n"
        f"{toolstring}"
    )

    msgs.append(
        {
            "role": "system",
            "content": system_pr.getvalue()
        }
    )

    fc_questions: dict = {
        "What is the current date and time?": {
            "functionName": "GetTimeAndDateOfNow",
            "parameters": [],
        },
        "What is the current time?": {
            "functionName": "GetTimeAndDateOfNow",
            "parameters": [],
        },
        "What is the current date?": {
            "functionName": "GetTimeAndDateOfNow",
            "parameters": [],
        },
        "What is the weather in London?": {
            "functionName": "WeatherFromLocation",
            "parameters": [{"parameterName": "location", "allowed_value_type": "string", "parameterValue": "London"}],
        },
        "What is the weather at 41.881832, -87.640406?": {
            "functionName": "WeatherFromLatLon",
            "parameters": [
                {"parameterName": "latitude", "allowed_value_type": "float", "parameterValue": 41.881832},
                {"parameterName": "longitude", "allowed_value_type": "float", "parameterValue": -87.640406},
            ],
        },
        "current CEO of tesla": {  # who is the current ceo of tesla?
            "functionName": "WebSearch",
            "parameters": [{"parameterName": "query", "allowed_value_type": "string",
                            "parameterValue": "current CEO of tesla"}],
        },
        "What City is located at 41.881832, -87.640406?": {
            "functionName": "LatLonToCity",
            "parameters": [
                {"parameterName": "latitude", "allowed_value_type": "float", "parameterValue": 41.881832},
                {"parameterName": "longitude", "allowed_value_type": "float", "parameterValue": -87.640406},
            ],
        },
    }

    for fcq in fc_questions:
        msgs.append(
            {
                "role": "user",
                "content": fcq,
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "content": Helper.get_pretty_dict_json_no_sort(
                    fc_questions[fcq]
                ),
            }
        )

        if allow_backticked_json:
            msgs[-1]["content"] = "```json\n" + msgs[-1]["content"] + "\n```"

    return msgs


def function_call_generate(
    input_text: str,
    function_schema: dict | None = None,
    fc_call_generate_priming_history: list[Message] | None = None,
    toolstring: str | None = None,
    model: str = "llama3:instruct",
    evict: bool = False,
    temperature: float = 0,
    top_k: int = 40,
    top_p: float = 0.9,
    num_predict: int = 2048,
    seed: int = 0,
    max_retries_json_response_parse: int = 3,
    print_msgs: bool = False,
    print_response: bool = False,
    ollama_return_format: Literal["", "json"] = "json",
    allow_backticked_json: bool = False,
    print_content_json: bool = False,
    print_options: bool = False
) -> tuple[dict, dict] | None:
    """
    Generates text based on the input using a pre-trained language model with various parameters for customization of output.

    :param print_options:
    :param print_content_json:
    :param allow_backticked_json:
    :param ollama_return_format:
    :param input_text: The initial string that will be used as context or prompt to generate subsequent text.
    :type input_text: str
    :param function_schema: Optional schema defining the structure and expected behavior of this function's inputs/outputs, defaults to None.
    :type function_schema: dict | None
    :param fc_call_generate_priming_history: A history list containing previous messages that can influence text generation (e.g., for maintaining context), optional and defaults to None. Default is [].
    :type fc_call_generate_priming_history: list[Message] | None, default []
    :param toolstring: Optional string providing additional information or metadata about the function call that can be used within this session (e.g., for logging purposes), defaults to None. Default is [].
    :type toolstring: str | None, default None
    :param model: The name of the pre-trained language model to use ('llama3:instruct' by default). This parameter can be used if multiple models are available and you want to specify which one should run. Default is "llama3:instruct".
    :type model: str, default "llama3:instruct"
    :param evict: A boolean flag indicating whether or not the priming history list fc_call_generate_priming_history should be cleared after each function call (defaults to False). Default is False.
    :type evict: bool, default False
    :param temperature: Controls randomness in text generation; lower values result in less surprising outputs that stick closer to the input context while higher temperatures increase diversity and creativity of responses but may also introduce more errors (default 0). Default is 0.
    :type temperature: float, default 0
    :param top_k: The number of words from the end of a list sorted by their probability to be sampled next in text generation; this parameter helps control diversity and creativity while avoiding repetition (defaults to 40). Default is 40.
    :type top_k: int, default 40
    :param top_p: The cumulative probability used for sampling the next word during text generation that ensures high-probability words are sampled while allowing lower-probability but still likely alternatives (defaults to 0.9). Default is 0.9.
    :type top_p: float, default 00.9
    :param num_predict: The number of tokens/words the function will generate in response; this parameter can be used for controlling output length and complexity (defaults to 2048). Default is 2048.
    :type num_predict: int, default 2048
    :param seed: A random number generator seed that ensures reproducibility of results across different runs with the same input parameters; this parameter can be used for debugging and testing purposes (defaults to 0). Default is 0.
    :type seed: int, default 0
    :param max_retries_json_response_parse:
    :param print_response: asd
    :param print_msgs: asd

    The function uses a pre-trained language model specified by 'model' to generate text based on 'input_text'. It allows customization of the generation process through various parameters such as temperature and top_k/top_p for sampling. If provided with priming history in fc_call_generate_priming_history, this context is used during generation; otherwise, it defaults to an empty list which means no prior conversation will influence text output unless specified by the user.

    The function returns a tuple of two dictionaries: one containing metadata about the generated outputs and another with actual generated texts if any are produced (the latter may be None). If 'evict' is set to True, it clears fc_call_generate_priming_history after each call. This can help in maintaining a clean context for subsequent calls within an interactive session or when running multiple independent generations without carrying over history from one generation to the next.

    Note: The function assumes that there is some form of language model available and accessible through this interface, which should be capable of handling these parameters appropriately during text generation tasks. It's important for users to ensure their environment has access to such a pre-trained model before invoking the function.

    :return: A tuple containing metadata dictionary with keys 'metadata_info', and generated texts in another dict under key 'generated_text'. Both are None if no text is produced or relevant data isn't available at this time (e.g., during testing phases). Defaults to None for both elements of the returned tuple when not applicable, which means that either nothing was generated yet or there were issues with generation and it should be investigated further before proceeding.
    :rtype: tuple[dict, dict] | None
    """

    if not toolstring:
        toolstring = TOOLSSTRING

    if not function_schema:
        function_schema = FUNCTION_SCHEMA

    if not fc_call_generate_priming_history:
        fc_call_generate_priming_history = _get_fc_call_generate_priming_history(
            function_schema=function_schema,
            toolstring=toolstring,
            allow_backticked_json=allow_backticked_json
        )

    # logger.debug(f"function_prompt:\n{msgs[0]['content']}")

    previous_responses: list[dict] = []
    for i in range(max_retries_json_response_parse):
        response: dict | None = None
        try:
            response = ask_ollama_chat(
                streamed=False,
                system_prompt=None,
                prompt=input_text,
                msg_history=fc_call_generate_priming_history,
                model=model,
                evict=evict,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_predict=num_predict,
                seed=seed,
                print_msgs=print_msgs,
                print_response=print_response,
                return_format=ollama_return_format,
                print_options=print_options,
                max_tries_ollama_done_response=2
            )

            # logger.debug(Helper.get_pretty_dict_json_no_sort(response))

            response["try_num"] = i + 1

            cr: str = response["message"]["content"]
            content_json: dict

            if allow_backticked_json and cr.find("```json") >= 0:
                # content_json = _parse_json_backticked(cr)[0]
                content_json = _parse_json_backticked(cr)
            else:
                content_json = json.loads(cr)

            # del response["message"]["content"]

            if print_content_json:
                logger.debug(Helper.get_pretty_dict_json_no_sort(content_json))

            return response, content_json
        except Exception as e:
            if response:
                logger.error(str(e))
                logger.debug(Helper.get_pretty_dict_json_no_sort(response))
                previous_responses.append(response)

            if i >= max_retries_json_response_parse - 1:
                logger.exception(e)

    return None


def _parse_json_backticked(resp: str) -> dict:  # list[dict]:
    # StructuredChatOutputParser
    #
    # https://github.com/langchain-ai/langchain/issues/8357
    #
    # match = re.search(r"```(json)?(.*?)```", json_string, re.DOTALL)
    # action_match = re.search(r"```(.*?)```", text, re.DOTALL)

    # logger.debug(f"{resp=}")
    # https://github.com/langchain-ai/langchain/pull/5037/files
    scone: str | None

    w: int = resp.find("```json")
    # logger.debug(f"{w=}")
    #if resp.find("```json") >= 0:
    scone = resp[w+len("```json"):]
    scone = scone[:scone.find("```")]

    scone = scone.strip()

    # logger.debug(f"{scone=}")

    return json.loads(scone)

    # sections: List[str] = resp.split("```json")
    # cleaned_sections: List[str] = [
    #     section[:section.find("```")].rstrip("```") for section in sections
    # ]
    #
    # responses: list[dict] = []
    # for output in cleaned_sections:
    #     # logger.debug(f"SECTION:\n{output}")
    #     try:
    #         response = json.loads(output)
    #         responses.append(response)
    #     except json.JSONDecodeError as e:
    #         # logger.error(str(e))
    #         pass  # Continue to the next section
    #
    # # logger.debug(Helper.get_pretty_dict_json_no_sort(responses))
    # return responses


def embeddings(prompt: str, embed_model: str = "nomic-embed-text:latest", num_ctx: int|None = 8192) -> Sequence[float|int]:
    # nomic num_ctx: 8192
    # mxbai num_ctx: 512

    options: dict|None

    if num_ctx is None:
        options = None
    else:
        options = {"num_ctx": num_ctx}

    ret: Mapping[str, Sequence[float]] = OLLAMA_CLIENT.embeddings(model=embed_model, prompt=prompt, options=options)

    return ret["embedding"]

def purge_model(model: str):
    OLLAMA_CLIENT.generate(model=model, keep_alive=0)  # unloading model from vram


def print_me(
    *values: object,
    end: str | None = "\n",
    flush: bool = False,
    sep: str | None = " ",
    file: "SupportsWrite[str] | None" = sys.stdout,
) -> None:
    print(*values, flush=flush, end=end, sep=sep, file=file)


def render_prompt_template(template_basename: str, md_or_plaintext: str, ollama_model: str|None = None, lang: Literal["en", "de"] = "en", template_type: TemplateType = TemplateType.plain) -> str:
    fp: Path = Path(TEMPLATEDIRPATH, template_type.value)
    fp = Path(fp, f"{template_basename}_{lang}_{template_type.value}.jinja")

    with open(fp) as file_:
        template = Environment(loader=FileSystemLoader(fp.parent), trim_blocks=True, lstrip_blocks=True).from_string(file_.read())

    values: dict = {
        "lang": lang,
        "md_or_plaintext": md_or_plaintext,
        "ollama_model": ollama_model,
        "template_type": template_type.value
    }

    ret: str = template.render(values)
    return ret

get_create_redo_prompt = partial(render_prompt_template, template_basename="get_redo_prompt")
get_create_summary_prompt = partial(render_prompt_template, template_basename="get_summary")
get_create_outline_prompt = partial(render_prompt_template, template_basename="get_outline")
get_reformat_and_semantically_markup_prompt = partial(render_prompt_template, template_basename="get_reformat_and_markup")

# class MyBaseLoader(BaseLoader):
#     def __init__(self):
#         super(MyBaseLoader, self).__init__()
#
#     def get_source(self, environment, template):
#         print(f"{type(environment)=} {environment=}")
#         print(f"{type(template)=} {template=}")
#
#         path = Path(_templatedirpath, template)
#         print(f"{type(path)=} {path=}  {path.exists()=}")
#
#         if not exists(path):
#             raise TemplateNotFound(template)
#         mtime = getmtime(path)
#         with open(path) as f:
#             source = f. read()
#         return source, path, lambda: mtime == getmtime(path)




def main():
    # compare_embeds()


    k={"h1": 1, "h2": 2}
    for kn in k:
        print(kn)

    from arley.dbobjects.ragdoc import DocTypeEnum
    for dt in DocTypeEnum:
        print(f"{type(dt)=} {dt=} {str(dt)=}")


    if len(sys.argv) > 1 and sys.argv[1] == "list":
        get_available_models()
        if threading.current_thread().name == "MainThread":
            exit(0)

    if len(sys.argv) > 2 and sys.argv[1] == "purge":
        OLLAMA_CLIENT.generate(model=sys.argv[2], keep_alive=0)  # unloading model from vram
        if threading.current_thread().name == "MainThread":
            exit(0)

    # TODO add some cmdline helpers here ?!


if __name__ == "__main__":
    logger.debug("__main__")

    main()