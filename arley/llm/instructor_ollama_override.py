import os
import sys
from datetime import datetime
from typing import Optional, Literal

import httpx
import json

from httpx import ByteStream
from openai import OpenAI

import instructor

_HOST: str = "http://127.0.0.1:11434"
_OLLAMA_MODEL: str = "hermes3:8b-llama3.1-fp16"
_OLLAMA_OPTIONS: dict = {
    "num_gpu": -1,
    "num_ctx": 65_536,
    "num_predict": -1
}

_OLLAMA_FORMAT_REQUEST: Literal['json']|None = None

try:
    import arley.config as config
    from loguru import logger

    _HOST = config.OLLAMA_HOST
    _OLLAMA_MODEL = config.settings.ollama.ollama_model
    _OLLAMA_OPTIONS = config.get_ollama_options(model=_OLLAMA_MODEL)
except ImportError as ie:
    print(f"IMPORT FAIL :: {ie=}", file=sys.stderr)

    from loguru import logger
    os.environ["LOGURU_LEVEL"] = os.getenv("LOGURU_LEVEL", "DEBUG")  # standard is DEBUG
    logger.remove()  # remove default-handler
    logger_fmt: str = "<g>{time:HH:mm:ssZZ}</> | <lvl>{level}</> | <c>{module}::{extra[classname]}:{function}:{line}</> - {message}"

    logger.add(sys.stderr, level=os.getenv("LOGURU_LEVEL"), format=logger_fmt)  # type: ignore # TRACE | DEBUG | INFO | WARN | ERROR |  FATAL
    logger.configure(extra={"classname": "None"})


# logger.disable(__name__)

logger.debug(f"{_HOST=}")
logger.debug(f"{_OLLAMA_MODEL=}")
logger.debug(f"{_OLLAMA_OPTIONS=}")

# https://gist.github.com/vroomfondel/eb1fb4ac3319b22f9dc3d9ff658ce0c9

class InstructorOpenAIOllamaOverride:  # metaclass=Singleton):
    logger = logger.bind(classname=__qualname__)

    def __init__(self, host: str, options: dict, print_request: bool, print_response: bool, print_http_request: bool, print_http_response: bool, think_flag: bool|None = None):
        self.host = host
        self.options = options
        self.print_request = print_request
        self.print_response = print_response
        self.print_http_request = print_http_request,
        self.print_http_response = print_http_response
        self.think_flag = think_flag

    @classmethod
    def get_instructor_client(cls,
                              host: str=_HOST,
                              options: Optional[dict] = None,
                              print_request: bool = False,
                              print_response: bool = False,
                              print_http_request: bool = False,
                              print_http_response: bool = False,
                              think_flag: bool|None = None) -> instructor.Instructor:
        if options is None:
            options = _OLLAMA_OPTIONS

        ioao: InstructorOpenAIOllamaOverride = InstructorOpenAIOllamaOverride(
            host=host,
            options=options,
            print_request=print_request,
            print_response=print_response,
            print_http_request=print_http_request,
            print_http_response=print_http_response,
            think_flag=think_flag
        )

        meclient = httpx.Client(
            event_hooks={
                'request': [
                    # partial(cls.modify_request, host = host, options = options)
                    ioao.modify_request
                ],
                'response': [
                    # partial(cls.modify_respone, host = host, options = options)
                    ioao.modify_response
                ]
            }
        )

        client: instructor.Instructor = instructor.from_openai(
            OpenAI(
                http_client=meclient,
                base_url=f"{host}/BLARGHNOTEXISTFAILCHECK",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON
        )

        return client

    def modify_request(self, request: httpx.Request) -> None:
        logger = self.__class__.logger
        if self.print_http_request:
            logger.debug(f"\nRequest event hook MODIFY: {request.method} {request.url} - Waiting for response")
            logger.debug(f"{type(request)=} {request=}")
            logger.debug(f"{type(request._content)=} {request.content=}")
            logger.debug(f"{type(request.stream)=} {request.stream=}")

        post_content: dict = json.loads(request.content)
        if self.print_request:
            logger.debug(f"REQ_CONTENT_OLD_PARSED: {json.dumps(post_content, indent=2, sort_keys=False, default=str)}")

        post_content_new: dict = {
            "model": post_content["model"],
            "messages": post_content["messages"],
            "tools": [],
            "stream": False,
            "options": self.options,
            "keep_alive": 300,
        }

        if self.think_flag is not None:
            post_content_new["think"] = self.think_flag

        if _OLLAMA_FORMAT_REQUEST:
            post_content_new["format"] = _OLLAMA_FORMAT_REQUEST

        request.json = post_content_new  # type: ignore  # just to be sure...

        # rebuilding stream
        # .venv/lib/python3.12/site-packages/httpx/_transports/default.py
        content_type: str | None = request.headers.get("content-type")
        headers, stream = httpx._content.encode_request(
            content=None,
            data=None,
            files=None,
            json=post_content_new,
            boundary=httpx._models.get_multipart_boundary_from_content_type(
                content_type=content_type.encode(request.headers.encoding)
                if content_type
                else None
            ),
        )

        request.headers["Content-Length"] = headers["Content-Length"]

        request._prepare(headers)
        request.stream = stream
        # Load the request body, except for streaming content.

        request.__delattr__("_content")

        if isinstance(stream, ByteStream):
            request.read()

        # /rebuilding stream

        # import traceback
        # traceback.print_stack()

        if self.print_request:
            logger.debug(f"REQ_CONTENT_NEW_PARSED: {json.dumps(post_content_new, indent=2, sort_keys=False, default=str)}")

        if self.print_http_request:
            logger.debug(f"REQ_OLD {type(request.url)=} {request.url=}")
            request.url = httpx.URL(f"{self.host}/api/chat")  # could be necessary to actually check chat-mode ?!
            logger.debug(f"REQ_NEW {type(request.url)=} {request.url=}")

    def modify_response(self, response: httpx.Response) -> None:
        logger = self.__class__.logger

        response.read()
        resp_content: dict = json.loads(response.content)

        if self.print_response:
            logger.debug(f"RESPONSE_CONTENT BEFORE:\n{json.dumps(resp_content, indent=2, sort_keys=False, default=str)}")

        uni: int = int(datetime.fromisoformat(resp_content["created_at"]).timestamp())
        mimic_openai: dict = {
            "id": f"rambling-{uni}",
            "object": "chat.completion",
            "created": uni,
            "model": resp_content["model"],
            "system_fingerprint": "thumb",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": resp_content["message"]["content"],
                    },
                    "finish_reason": resp_content["done_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": resp_content["prompt_eval_count"],
                "completion_tokens": resp_content["eval_count"],
                "total_tokens": resp_content["prompt_eval_count"] + resp_content["eval_count"],
            },
        }

        if self.think_flag and "thinking" in resp_content["message"]:
            mimic_openai["choices"][0]["message"]["reasoning_content"] = resp_content["message"]["thinking"]

        response._content = json.dumps(mimic_openai).encode()

        if self.print_response:
            logger.debug(f"RESPONSE_CONTENT MODIFIED:\n{json.dumps(mimic_openai, indent=2, sort_keys=False, default=str)}")


def main() -> None:
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    instructor_client = InstructorOpenAIOllamaOverride.get_instructor_client(host=_HOST, options=_OLLAMA_OPTIONS, think_flag=True, print_request=True, print_response=True)

    resp, comp = instructor_client.create_with_completion(
        stream=False,
        model=_OLLAMA_MODEL,
        messages=[
            # {"role": "system", "content": "You are a pirate."},
            # {
            #     "role": "user",
            #     "content": "Tell me about the Harry Potter",
            # }
            {"role": "user", "content": "Create a user with a funny name"}
        ],
        response_model=User,
    )
    logger.debug(f"{type(comp)=}")
    logger.debug(comp.model_dump_json(indent=2))

    # logger.debug(f"{type(completion)=} {completion=}")
    logger.debug(f"{type(resp)=}")
    logger.debug(resp.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
