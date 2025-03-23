import json
from io import StringIO
from typing import Literal

import instructor
import openai
from ollama import Message
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field

from arley import Helper
from arley.config import OLLAMA_HOST, get_ollama_options, OLLAMA_MODEL

from arley.llm.instructor_ollama_override import InstructorOpenAIOllamaOverride

from loguru import logger

class LangDetect(BaseModel):
    type: Literal["string"] = "string"
    lang: Literal["de", "en"] = Field(description="either 'de' for detected language being german or 'en' for detected language being english.")

class LangInputSupplied(BaseModel):
    type: Literal["string"] = "string"
    input: str = Field(description="precisely provide the full input the user gave you for detection (and not more!)")

class LanguageGuessResponseSchema(BaseModel):
    language_detected: LangDetect
    input_supplied: LangInputSupplied





class LanguageGuesser:
    logger = logger.bind(classname=__qualname__)

    @classmethod
    def guess_language(cls,
                       input_text: str,
                       only_return_str: bool = True,
                       ollama_host: str = OLLAMA_HOST,
                       ollama_model: str = OLLAMA_MODEL,
                       ollama_options: dict|None = None,
                       print_msgs: bool = False,
                       print_http_request: bool = False,
                       print_response: bool = False,
                       print_http_response: bool = False,
                       max_retries: int = 3) -> str | tuple[str, dict, dict] | None:

        if ollama_options is None:
            ollama_options = get_ollama_options(model=ollama_model)

        # str | lang, response, json
        instructor_client: instructor.Instructor = InstructorOpenAIOllamaOverride.get_instructor_client(
            host=ollama_host,
            options=ollama_options,
            print_request=print_http_request,
            print_response=print_http_response
        )

        msgs: list[dict] = cls._get_guess_language_priming_history(new_content=input_text)
        if print_msgs:
            logger.debug(Helper.get_pretty_dict_json_no_sort(msgs))

        resp: LanguageGuessResponseSchema | None
        comp: ChatCompletion | None

        resp, comp = instructor_client.create_with_completion(
            stream=False,
            model=ollama_model,
            messages=msgs,
            response_model=LanguageGuessResponseSchema,
            max_retries=max_retries
        )

        if print_response:
            logger.debug(f"{type(comp)=}")
            logger.debug(comp.model_dump_json(indent=2))

            # logger.debug(f"{type(completion)=} {completion=}")
            logger.debug(f"{type(resp)=}")
            logger.debug(resp.model_dump_json(indent=2))

        if only_return_str:
            return resp.language_detected.lang

        return resp.language_detected.lang, comp.model_dump(), resp.model_dump()


    @staticmethod
    def _get_guess_language_priming_history(new_content: str | None = None) -> list[dict]:  # list[Message]:
        msgs: list[Message] = []

        system_pr: StringIO = StringIO()

        system_pr.write(
            f"You are a language detection engine.\n"
            f"Your will detect the language of the request a user supplies to you.\n"
            f"You will NOT try to answer to the request itself, but will only use the request supplied to you to detect the language from that.\n")
        # f'If I am unsure, I will just make an educated guess, but I will absolutely make sure to only answer with "lang" being either "en" or "de" under any circumstance.\n'

        msgs.append(
            {
                "role": "system",
                "content": system_pr.getvalue()
            }
        )

        inputs: list[str] = [
            f"Hallo, wie geht's?!",
            f"What a nice idea!"
        ]
        langs: list[str] = [
            "de",
            "en"
        ]

        for meinput, lang in zip(inputs, langs):
            msgs.append({"role": "user", "content": meinput})

            r1: LanguageGuessResponseSchema = LanguageGuessResponseSchema(
                language_detected=LangDetect(lang="de"),
                input_supplied=LangInputSupplied(input=meinput)
            )

            msgs.append(
                {
                    "role": "assistant",
                    "content": json.dumps(r1.model_dump(mode="json"), indent=2, default=str)
                }
            )

        if new_content:
            msgs.append(
                {
                    "role": "user",
                    "content": new_content
                }
            )

        return msgs



if __name__ == "__main__":
    langcode: str = LanguageGuesser.guess_language(
        # input_text="Warum larum wer sind sie?\n\nKannst Du auch mehrzeilig?",
        input_text="terrified terrier terry does not want to eat his chocolate.",
        only_return_str=True,
        ollama_host=OLLAMA_HOST,
        ollama_model=OLLAMA_MODEL,
        ollama_options=get_ollama_options(OLLAMA_MODEL),
        print_msgs=True,
        print_http_request=True,
        print_http_response=True,
        print_response=True,
        max_retries=0
    )

    print(f"{'#'*10} LANGCODE: {langcode}")
