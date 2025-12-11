import json
import time
from typing import Any, Mapping, Iterator, Literal

from ollama import Message

from arley.config import settings, OLLAMA_HOST, get_ollama_options
from arley import Helper
from arley.llm.language_guesser import LanguageGuesser
from arley.llm.ollama_adapter import _get_fc_call_generate_priming_history, function_call_generate, \
    ask_ollama_chat, purge_model, _parse_json_backticked

from loguru import logger

from arley.llm.ollama_tools import FUNCTION_SCHEMA, TOOLSSTRING, TOOLSLIST_OLLAMA_TOOL_STYLE, ToolBox


def add_and_update_metrics_in_place(
        results_dict: dict,
        eval_dict: dict,
        ollama_response: dict,
        this_eval_start: float,
        this_eval_end: float
) -> None:

    # logger.debug(f"RESPONSE(CONTENT_STRIPPED_OUT):\n{Helper.get_pretty_dict_json_no_sort(ollama_response)}")

    if not ollama_response["done"]:
        logger.debug(Helper.get_pretty_dict_json_no_sort(ollama_response))

    try:
        results_dict["total_duration_s"] = ollama_response["total_duration"] / 1_000_000_000
        results_dict["load_duration_s"] = ollama_response["load_duration"] / 1_000_000_000
        results_dict["prompt_eval_duration_s"] = ollama_response["prompt_eval_duration"] / 1_000_000_000
        results_dict["eval_duration_s"] = ollama_response["eval_duration"] / 1_000_000_000
        results_dict["run_time_s"] = this_eval_end - this_eval_start

        eval_dict["total_duration_s"] += results_dict["total_duration_s"]
        eval_dict["load_duration_s"] += results_dict["load_duration_s"]
        eval_dict["prompt_eval_duration_s"] += results_dict["prompt_eval_duration_s"]
        eval_dict["eval_duration_s"] += results_dict["eval_duration_s"]
    except Exception as e:
        logger.error("SKIPPERD: "+str(e))

def fc_compare(desired: dict[str, Any], got: dict[str, Any]) -> bool:
    # logger.debug(f"DESIRED:\n{Helper.get_pretty_dict_json_no_sort(desired)}\n"
    #              f"GOT:\n{Helper.get_pretty_dict_json_no_sort(got)}")

    if desired["functionName"] != got["functionName"]:
        return False

    if "parameters" not in got:
        return False

    fn: str = desired["functionName"]
    # logger.debug(f"{fn}")

    params: list[dict] = desired["parameters"]
    for i, param in enumerate(params):
        # logger.debug(f"{i=} {param=}")

        if desired["parameters"][i] != got["parameters"][i]:
            if "parameterName" not in got["parameters"][i]:
                return False
            if "parameterValue" not in got["parameters"][i]:
                return False
            if "allowed_value_types" not in got["parameters"][i]:
                return False

            if desired["parameters"][i]["parameterName"] != got["parameters"][i]["parameterName"]:
                return False

            if desired["parameters"][i]["allowed_value_type"] != got["parameters"][i]["allowed_value_type"]:
                return False

            if fn == "WebSearch":
                if desired["parameters"][i]["parameterValue"].lower() == got["parameters"][i]["parameterValue"].lower():
                    # logger.debug(f"OVERRIDE: {got["parameters"][i]["parameterValue"]}")
                    continue

            return False

    return True


def fc_eval_func(model: str, ret: dict | None,
                 ollama_return_format: Literal["", "json"] = "json",
                 allow_backticked_json: bool = False) -> dict:
    fc_eval_start = time.perf_counter()
    fc_eval: dict = {}

    if ret:
        ret["fc_eval"] = fc_eval

    fc_eval["total_duration_s"] = 0.0
    fc_eval["load_duration_s"] = 0.0
    fc_eval["prompt_eval_duration_s"] = 0.0
    fc_eval["eval_duration_s"] = 0.0

    fc_results: dict = {}
    fc_eval["results"] = fc_results

    fc_questions: dict = {}
    # just recycling the priming questions...
    msgs: list[Message] = _get_fc_call_generate_priming_history(
        function_schema=FUNCTION_SCHEMA,
        toolstring=TOOLSSTRING,
        allow_backticked_json=allow_backticked_json
    )

    logger.info(f"SYSTEM_PROMPT:\n{msgs[0]["content"]}")

    for i in range(1, len(msgs), 2):
        msg: Message = msgs[i]
        msgn: Message = msgs[i + 1]

        fc_questions[msg["content"]] = msgn["content"]

    # logger.debug(f"FC QUESTIONS:\n{Helper.get_pretty_dict_json_no_sort(fc_questions)}")

    for fctest_question in fc_questions:
        logger.debug(f"Q: {fctest_question}")

        fc_success: bool = False

        fc_results[fctest_question] = {}
        this_fc_eval_start = time.perf_counter()

        fc: tuple[dict, dict] | None = function_call_generate(
            input_text=fctest_question,
            model=model,
            ollama_return_format=ollama_return_format,
            allow_backticked_json=allow_backticked_json,
            print_content_json=True,
            num_predict=-1,
            print_msgs=False,
            print_response=True,
            print_options=False
        )

        this_fc_eval_end = time.perf_counter()

        # logger.debug(f"QUESTION WAS: \"{fctest_question}\"")
        # logger.debug(f"fc_questions[fctest_question]: \"{fc_questions[fctest_question]}\"")

        desired_result: dict
        if allow_backticked_json:
            # desired_result = _parse_json_backticked(fc_questions[fctest_question])[0]
            desired_result = _parse_json_backticked(fc_questions[fctest_question])
        else:
            desired_result = json.loads(fc_questions[fctest_question])


        if fc:
            ollama_response, content = fc

            logger.debug(Helper.get_pretty_dict_json_no_sort(content))

            add_and_update_metrics_in_place(
                results_dict=fc_results[fctest_question],
                eval_dict=fc_eval,
                ollama_response=ollama_response,
                this_eval_start=this_fc_eval_start,
                this_eval_end=this_fc_eval_end
            )

            # logger.debug(f"RESPONSE(CONTENT_STRIPPED_OUT):\n{Helper.get_pretty_dict_json_no_sort(response)}")
            # logger.debug(f"CONTENT:\n{Helper.get_pretty_dict_json_no_sort(content)}")

            #     "total_duration": 1228756388,
            #     "load_duration": 15147725,
            #     "prompt_eval_count": 943,
            #     "prompt_eval_duration": 12842000,
            #     "eval_duration": 699493000


            # if desired_result == content:
            if fc_compare(desired_result, content):
                fc_success = True

                logger.debug("CALLING FUNCTION :::: WOOOOT")
                ToolBox.get_instance().execute_tool_function(content["functionName"], content["parameters"])
            else:
                logger.debug(
                    f"FC MISMATCH:\nGOT:\n{Helper.get_pretty_dict_json_no_sort(content)}\n\tVS\nDESIRED:\n{Helper.get_pretty_dict_json_no_sort(desired_result)}")
        else:
            logger.error("FC_RESP IS NONE")

        fc_results[fctest_question]["success"] = fc_success

    fc_eval_end = time.perf_counter()
    # fc_results["ALL"]["run_time_s"] = fc_eval_end - fc_eval_start

    return fc_eval


def lang_eval(
        model: str,
        ret: dict | None,
        ollama_return_format: Literal["", "json"] = "json",
        allow_backticked_json: bool = False) -> dict:

    lang_eval_start = time.perf_counter()
    lang_eval: dict = {}

    if ret:
        ret["lang_eval"] = lang_eval

    lang_eval["total_duration_s"] = 0.0
    lang_eval["load_duration_s"] = 0.0
    lang_eval["prompt_eval_duration_s"] = 0.0
    lang_eval["eval_duration_s"] = 0.0

    lang_results: dict = {}
    lang_eval["results"] = lang_results

    #     "total_duration": 1228756388,
    #     "load_duration": 15147725,
    #     "prompt_eval_count": 943,
    #     "prompt_eval_duration": 12842000,
    #     "eval_duration": 699493000

    lang_questions: dict = {
        "Ich habe keine Ahnung, was hier abgeht.\n\nWer sind sie?": "de",
        "Kalter Kaffee auf Eis ohne Milch Kakao schaumig.": "de",
        "Once upon a time, there were four little rabbits.": "en",
        "Whatever floats your boat, Crumbledore!": "en",
    }

    for langq in lang_questions:
        lang_success = False

        lang: str
        ollama_response: dict
        lang_detect_content: dict

        desired_result: str = lang_questions[langq]

        lang_results[langq] = {}
        this_lang_eval_start = time.perf_counter()


        res = LanguageGuesser.guess_language(input_text=langq, only_return_str=False, ollama_host=OLLAMA_HOST,
                                             ollama_model=model, ollama_options=get_ollama_options(model),
                                             print_msgs=True, print_response=True, print_http_response=False,
                                             print_http_request=False, max_retries=3)

        if res:
            lang, ollama_response, lang_detect_content = res  # type: ignore
        else:
            raise RuntimeError("WOOHOO")

        this_lang_eval_end = time.perf_counter()

        add_and_update_metrics_in_place(
            lang_results[langq],
            lang_eval,
            ollama_response,
            this_lang_eval_start,
            this_lang_eval_end
        )

        # logger.debug(Helper.get_pretty_dict_json_no_sort(ollama_response)
        logger.debug(Helper.get_pretty_dict_json_no_sort(lang_detect_content))
        logger.debug(f"{'*' * 80}")
        if lang == desired_result:
            lang_success = True
            logger.debug(f"{lang} SUCCESS!!! for {model}")
        else:
            logger.debug(f"{lang} FAIL!!! for {model}")
        logger.debug(f"{'*' * 80}")

        lang_results[langq]["success"] = lang_success

    lang_eval_end = time.perf_counter()
    # lang_results[langq]["start_time"] = lang_eval_start
    # lang_results[langq]["end_time"] = lang_eval_end
    # lang_results["run_time_s"] = lang_eval_end - lang_eval_start

    return lang_eval



# TODO HT 20240717 INPLACE DETECTION OF FUNCTION-CALLS
# def fc_combine_eval(model: str, ret: dict | None) -> dict:
#     fc_eval_start = time.perf_counter()
#     fc_eval: dict = {}
#
#     if ret:
#         ret["fc_eval"] = fc_eval
#
#     fc_eval["total_duration_s"] = 0.0
#     fc_eval["load_duration_s"] = 0.0
#     fc_eval["prompt_eval_duration_s"] = 0.0
#     fc_eval["eval_duration_s"] = 0.0
#
#     fc_results: dict = {}
#     fc_eval["results"] = fc_results
#
#     fc_questions: dict = {}
#     # just recycling the priming questions...
#     msgs: list[Message] = _get_fc_call_generate_priming_history(function_schema=FUNCTION_SCHEMA, toolstring=TOOLSSTRING)
#     for i in range(1, len(msgs), 2):
#         msg: Message = msgs[i]
#         msgn: Message = msgs[i + 1]
#
#         fc_questions[msg["content"]] = msgn["content"]
#
#     # logger.debug(f"FC QUESTIONS:\n{Helper.get_pretty_dict_json_no_sort(fc_questions)}")
#
#     for fctest_question in fc_questions:
#         fc_success: bool = False
#
#         fc_results[fctest_question] = {}
#         this_fc_eval_start = time.perf_counter()
#
#         fc: tuple[dict, dict] | None = function_call_generate(
#             input_text=fctest_question,
#             model=model,
#             print_msgs=False,
#             print_response=False,
#             max_retries_json_response_parse=1,
#             # num_predict=2048
#         )
#
#         this_fc_eval_end = time.perf_counter()
#
#         desired_result: dict = json.loads(fc_questions[fctest_question])
#
#
#         logger.debug(f"QUESTION WAS: \"{fctest_question}\"")
#         if fc:
#             ollama_response, content = fc
#
#             add_and_update_metrics_in_place(
#                 results_dict=fc_results[fctest_question],
#                 eval_dict=fc_eval,
#                 ollama_response=ollama_response,
#                 this_eval_start=this_fc_eval_start,
#                 this_eval_end=this_fc_eval_end
#             )
#
#             # logger.debug(f"RESPONSE(CONTENT_STRIPPED_OUT):\n{Helper.get_pretty_dict_json_no_sort(response)}")
#             # logger.debug(f"CONTENT:\n{Helper.get_pretty_dict_json_no_sort(content)}")
#
#             #     "total_duration": 1228756388,
#             #     "load_duration": 15147725,
#             #     "prompt_eval_count": 943,
#             #     "prompt_eval_duration": 12842000,
#             #     "eval_duration": 699493000
#
#             if desired_result == content:
#                 fc_success = True
#
#                 logger.debug("CALLING FUNCTION :::: WOOOOT")
#                 execute_tool_function(content["functionName"], content["parameters"])
#             else:
#                 logger.debug(
#                     f"FC MISMATCH:\n{Helper.get_pretty_dict_json_no_sort(content)}\n\tVS\n{Helper.get_pretty_dict_json_no_sort(desired_result)}")
#         else:
#             logger.error("FC_RESP IS NONE")
#
#         fc_results[fctest_question]["success"] = fc_success
#
#     fc_eval_end = time.perf_counter()
#     # fc_results[fctest_question]["run_time_s"] = fc_eval_end - fc_eval_start
#
#     return fc_eval



def basic_evalrun(model: str = "llama3:latest", lang_ollama_return_format: Literal["", "json"] = "json", lang_allow_backticked_json: bool = False,
                  fc_ollama_return_format: Literal["", "json"] = "json", fc_allow_backticked_json: bool = False) -> dict:
    ret: dict = {}

    start_time = time.perf_counter()
    logger.debug(f"{'*' * 80}")
    logger.debug(f"model={model}")
    logger.debug(f"{'*' * 80}")

    # purge_model(model=model)

    # preload model
    resp: dict | Mapping[str, Any] | Iterator[Mapping[str, Any]] = ask_ollama_chat(
        system_prompt="you are a very smart ai engine.",
        model=model,
        print_response=True,
        prompt=""  # empty prompt to preload model  # https://github.com/ollama/ollama/issues/2431
    )

    preload: dict = {}
    ret["preload"] = preload
    preload["total_duration_s"] = resp["total_duration"] / 1_000_000_000  # type: ignore
    preload["load_duration_s"] = resp["load_duration"] / 1_000_000_000  # type: ignore
    preload["prompt_eval_duration_s"] = resp["prompt_eval_duration"] / 1_000_000_000  # type: ignore
    preload["eval_duration_s"] = resp["eval_duration"] / 1_000_000_000  # type: ignore

    ##################
    ### lang_eval_stuff
    lang_eval_dict: dict = lang_eval(
        model=model,
        ret=ret,
        ollama_return_format=lang_ollama_return_format,
        allow_backticked_json=lang_allow_backticked_json
    )
    lang_results: dict = lang_eval_dict["results"]

    # logger.debug(Helper.get_pretty_dict_json_no_sort(lang_eval_dict))

    logger.debug(f"{'*' * 80}")
    success: int = 0
    for langresult_q in lang_results:
        logger.debug(langresult_q)
        langresult_b = lang_results[langresult_q]["success"]
        success += 1 if langresult_b else 0

    logger.debug(f"LANG OVERALL_SUCCESS for {model=}: {(success*100/len(lang_results)):.2f}%")


    ##################
    ### fc test stuff
    fc_eval_dict: dict = fc_eval_func(
        model=model,
        ret=ret,
        ollama_return_format=fc_ollama_return_format,
        allow_backticked_json=fc_allow_backticked_json
    )
    fc_results: dict = fc_eval_dict["results"]

    # logger.debug(Helper.get_pretty_dict_json_no_sort(fc_eval_dict))

    success = 0
    for fc_result_q in fc_results:
        fc_result_b = fc_results[fc_result_q]["success"]
        success += 1 if fc_result_b else 0

    logger.debug(f"FC OVERALL_SUCCESS for {model=}: {(success*100/len(fc_results)):.2f}%")

    logger.debug(f"{'*' * 80}")
    end_time = time.perf_counter()
    run_time = end_time - start_time
    logger.debug(f"\tFinished {model} in {round(run_time, 3)}s")
    logger.debug(f"{'*' * 80}\n")

    ret["run_time_s"] = run_time

    return ret


def main() -> dict[str, dict[str, Any]]:
    # "mixtral:8x22b-text"
    # "nous-hermes2-mixtral:8x7b"
    # "mixtral:8x22b-instruct"
    # "mixtral:text"
    # "gemma2:27b"
    # "nomic-embed-text:latest"
    # "llama3-gradient:instruct"
    # "mixtral:8x7b-instruct-v0.1-q8_0"
    # "llama3:latest"
    # "mixtral:latest",


    all_results: dict[str, dict[str, Any]] = {}
    for i in [
        # "llama3.1:70b-instruct-q3_K_M",
        "llama3.1:latest",
        # "llama3-chatqa:8b-v1.5-q8_0",
        # "llama3-gradient:8b-instruct-1048k-q8_0",
        # "llama3:latest",
        # "gemma2:27b",
        # "llama3-gradient:instruct",  #"llama3-gradient:8b-instruct-1048k-q8_0",  # "llama3-gradient:instruct",
        # "nous-hermes2-mixtral:8x7b",
        # "hermes3:8b-llama3.1-fp16",
        # "mixtral:latest"
        # "mixtral:8x7b-instruct-v0.1-q8_0"
    ]:


        lang_allow_backticked_json = False
        lang_ollama_return_format: Literal["", "json"] = "json"

        fc_allow_backticked_json = False
        fc_ollama_return_format: Literal["", "json"] = "json"


        if ALLOW_JSON_RETURN_FOR_MIXTRAL and i.find("mixtral") >= 0:
            lang_allow_backticked_json = True
            lang_ollama_return_format = ""

            fc_allow_backticked_json = True
            fc_ollama_return_format = ""


        this_model_results: dict = basic_evalrun(
            model=i,
            lang_ollama_return_format=lang_ollama_return_format,
            lang_allow_backticked_json=lang_allow_backticked_json,

            fc_ollama_return_format=fc_ollama_return_format,
            fc_allow_backticked_json=fc_allow_backticked_json
        )

        all_results[i] = this_model_results

    logger.debug(Helper.get_pretty_dict_json_no_sort(all_results))

    return all_results


if __name__ == "__main__":
    ALLOW_JSON_RETURN_FOR_MIXTRAL: bool = True
    main()

