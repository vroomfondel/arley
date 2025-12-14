import json
import os
import sys
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Type

import pytz
from cachetools import TTLCache, cached
from pydantic import BaseModel, Field, HttpUrl, PostgresDsn, RootModel

_CONFIGDIRPATH: Path = Path(__file__).parent.resolve()
_CONFIGDIRPATH = Path(os.getenv("ARLEY_CONFIG_DIR_PATH")) if os.getenv("ARLEY_CONFIG_DIR_PATH") else _CONFIGDIRPATH  # type: ignore

_CONFIGPATH: Path = Path(_CONFIGDIRPATH, "config.yaml")
_CONFIGPATH: Path = Path(os.getenv("ARLEY_CONFIG_PATH")) if os.getenv("ARLEY_CONFIG_PATH") else _CONFIGPATH  # type: ignore

_CONFIGLOCALPATH: Path = Path(_CONFIGDIRPATH, "config.local.yaml")
_CONFIGLOCALPATH = Path(os.getenv("ARLEY_CONFIG_LOCAL_PATH")) if os.getenv("ARLEY_CONFIG_LOCAL_PATH") else _CONFIGLOCALPATH  # type: ignore


from loguru import logger
from pydantic_settings import (BaseSettings, DotEnvSettingsSource,
                               EnvSettingsSource, InitSettingsSource,
                               PydanticBaseSettingsSource, SettingsConfigDict,
                               YamlConfigSettingsSource)

# https://buildmedia.readthedocs.org/media/pdf/loguru/latest/loguru.pdf
os.environ["LOGURU_LEVEL"] = os.getenv("LOGURU_LEVEL", "DEBUG")  # standard is DEBUG
logger.remove()  # remove default-handler
logger_fmt: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>::<cyan>{extra[classname]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
# logger_fmt: str = "<g>{time:HH:mm:ssZZ}</> | <lvl>{level}</> | <c>{module}::{extra[classname]}:{function}:{line}</> - {message}"

def _loguru_skiplog_filter(record: dict) -> bool:
    return not record.get("extra", {}).get("skiplog", False)

logger.add(sys.stderr, level=os.getenv("LOGURU_LEVEL"), format=logger_fmt, filter=_loguru_skiplog_filter)  # type: ignore # TRACE | DEBUG | INFO | WARN | ERROR |  FATAL
logger.configure(extra={"classname": "None", "skiplog": False})


logger.info(f"EFFECTIVE CONFIGPATH: {_CONFIGPATH}")
logger.info(f"EFFECTIVE CONFIGLOCALPATH: {_CONFIGLOCALPATH}")


# https://docs.pydantic.dev/latest/concepts/pydantic_settings/

# alias in settings not correctly handled for pydantic v2
# https://github.com/pydantic/pydantic/issues/8379


class TemplateType(StrEnum):
    plain = auto()
    xml = auto()
    plain_chat = auto()
    xml_chat = auto()


class OllamaPrimingMessage(BaseModel):
    role: Literal["system", "assistant", "user"]
    lang: Literal["en", "de"]
    content: str


class Folders(BaseModel):
    old: str = Field(default="WORKED")
    cur: str = Field(default="WORKING")
    rej: str = Field(default="REJECTED")
    err: str = Field(default="FAILED")


class Redis(BaseModel):
    host: str = Field(default="127.0.0.1")
    host_in_cluster: Optional[str] = Field(default=None)
    port: int = Field(default=6379)


class Chromadb(BaseModel):
    host: str = Field(default=os.getenv("CHROMADB_HOST", "127.0.0.1"))
    host_in_cluster: Optional[str] = Field(
        default=os.getenv("CHROMADB_HOST_CLUSTER", "chromadb.chromadb.svc.cluster.local")
    )
    port: int = Field(default=int(os.getenv("CHROMADB_PORT", "8000")))
    http_auth_user: Optional[str] = Field(default=None)
    http_auth_pass: Optional[str] = Field(default=None)
    default_collectionname: Optional[str] = Field(default=os.getenv("CHROMADB_DEFAULT_COLLECTIONNAME", "arley"))
    ollama_embed_model: Optional[str] = Field(
        default=os.getenv("CHROMADB_OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
    )


class Ollama(BaseModel):
    host: str = Field(default=os.getenv("OLLAMA_BASE_HOST", "127.0.0.1"))
    port: int = Field(default=int(os.getenv("OLLAMA_PORT", "11434")))
    host_in_cluster: Optional[str] = Field(
        default=os.getenv("OLLAMA_BASE_HOST_CLUSTER", "ollama.ollama.svc.cluster.local")
    )
    ollama_model: str = Field(default="llama3.1:latest")
    ollama_embed_model: str = Field(default="nomic-embed-text:latest")
    ollama_function_calling_model: str = Field(default="llama3.1:latest")
    ollama_guess_language_model: str = Field(default="llama3.1:latest")
    ollama_priming_messages: list[OllamaPrimingMessage] = Field(default_factory=list)


class Postgresql(BaseModel):
    host: str = Field(default="127.0.0.1")
    host_in_cluster: Optional[str] = Field(default="postgresql.postgresql.svc.cluster.local")
    username: str = Field(default="postgres")
    password: str = Field(default="empty")
    dbname: str = Field(default="postgres")
    port: int = Field(default=5432)
    # url: Optional[PostgresDsn] = Field(default=None)
    url: Optional[str] = Field(default=None)


class EmailSettings(BaseModel):
    privileged_sender: Optional[str] = Field(default=None)
    mailaddress: str
    mailuser: str
    mailpassword: str
    imapserver: str
    imapport: int
    smtpserver: str
    smtpport: int
    folders: Folders
    alloweddomains: list[str]


# TODO: HT20240912 -> make only_contracts-stuff obsolete by introducing the possibility to define lists of what to include
class ArleyAug(BaseModel):
    template_version: str = Field(default="v2")
    unified: bool = Field(default=False)
    per_item: bool = Field(default=True)
    num_docs: int = Field(default=5)
    template_type: TemplateType = Field(default=TemplateType["xml_chat"])
    only_contracts: bool = Field(default=False)
    lang_filter: bool = Field(default=True)
    first_request_include_aug: bool = Field(default=True)
    first_request_n_aug_results: int = Field(default=1)
    first_request_unified: bool = Field(default=True)
    first_request_per_item: bool = Field(default=False)
    first_request_template_type: TemplateType = Field(default=TemplateType["xml_chat"])
    first_request_aug_only_contracts: bool = Field(default=True)
    first_request_aug_lang_filter: bool = Field(default=True)


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        populate_by_name=True,
        # env_prefix="TAS_",
        case_sensitive=False,
        yaml_file_encoding="utf-8",
        extra="ignore",  # ignore | forbid | allow
        protected_namespaces=(),
        env_nested_delimiter="__",
        # alias_generator=AliasGenerator(
        #     validation_alias=to_camel,
        #     serialization_alias=to_pascal,
        # )
        yaml_file=[_CONFIGPATH, _CONFIGLOCALPATH],
    )

    arley_aug: ArleyAug
    emailsettings: EmailSettings
    redis: Redis
    postgresql: Postgresql
    ollama: Ollama
    chromadb: Chromadb
    timezone: str = Field(default="Europe/Berlin")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: InitSettingsSource,  # type: ignore
        env_settings: EnvSettingsSource,  # type: ignore
        dotenv_settings: DotEnvSettingsSource,  # type: ignore
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, YamlConfigSettingsSource(settings_cls)


def str2bool(v: str | bool) -> bool:
    if not v:
        return False

    if isinstance(v, bool):
        return v

    return v.lower() in ("yes", "true", "t", "1")


@cached(cache=TTLCache(maxsize=1, ttl=60))
def is_in_cluster() -> bool:
    sa: Path = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    if sa.exists() and sa.is_dir():
        return os.getenv("KUBERNETES_SERVICE_HOST") is not None
    return False


def log_settings() -> None:
    for k, v in os.environ.items():
        if k.startswith("PSQL_"):
            logger.info(f"ENV::{k}: {v}")
    logger.info(json.dumps(settings.model_dump(by_alias=True), indent=4, sort_keys=False, default=str))


def get_ollama_options(
    model: str,
    top_k: int | None = 40,
    top_p: float | None = 0.9,
    temperature: float | None = 0.8,
    seed: int | None = 0,
    num_predict: int | None = -1,
    repeat_penalty: float | None = 1.1,
) -> dict:

    num_ctx: int = get_num_ctx_by_model_name(model_name=model)

    # num_predict	Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)	int	num_predict 42
    # top_k	Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)	int	top_k 40
    # top_p	Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)

    # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

    options: dict = {}

    if top_k and top_k != 40:  # default: 40
        options["top_k"] = top_k
    if top_p and top_p != 0.9:  # default: 0.9
        options["top_p"] = top_p
    if temperature != 0.8:  # default 0.8
        options["temperature"] = temperature
    if seed and seed != 0:
        options["seed"] = seed
    if repeat_penalty and repeat_penalty != 1.1:
        options["repeat_penalty"] = repeat_penalty

    options["num_gpu"] = -1
    options["num_ctx"] = num_ctx
    options["num_predict"] = num_predict

    # options: dict = {
    #     "num_gpu": -1,  # -1: auto  | 0: none | int: num layers to offload to gpu
    #     "num_ctx": num_ctx,  # 4096 # default: 2048 https://github.com/ollama/ollama/blob/main/docs/faq.md
    #     "num_predict": num_predict,
    # "top_k": top_k,
    # "top_p": top_p,
    # "temperature": temperature,
    # "seed": seed,
    # "repeat_penalty": repeat_penalty,
    # "penalize_newline": penalize_newline,  # present in api-doc, but not in modelfile...
    # "mirostat": 1,
    # "repeat_last_n": 512,
    # "num_keep": 64,  # number of tokens to keep from initial prompt
    # "repeat_penalty": 1.5,
    # "seed": 123,
    # "temperature": 0

    # https://github.com/ollama/ollama/blob/main/docs/api.md
    # "num_keep": 5,
    #     "seed": 42,
    #     "num_predict": 100,
    #     "top_k": 20,
    #     "top_p": 0.9,
    #     "tfs_z": 0.5,
    #     "typical_p": 0.7,
    #     "repeat_last_n": 33,
    #     "temperature": 0.8,
    #     "repeat_penalty": 1.2,
    #     "presence_penalty": 1.5,
    #     "frequency_penalty": 1.0,
    #     "mirostat": 1,
    #     "mirostat_tau": 0.8,
    #     "mirostat_eta": 0.6,
    #     "penalize_newline": true,
    #     "stop": ["\n", "user:"],
    #     "numa": false,
    #     "num_ctx": 1024,
    #     "num_batch": 2,
    #     "num_gqa": 1,
    #     "num_gpu": 1,
    #     "main_gpu": 0,
    #     "low_vram": false,
    #     "f16_kv": true,
    #     "vocab_only": false,
    #     "use_mmap": true,
    #     "use_mlock": false,
    #     "rope_frequency_base": 1.1,
    #     "rope_frequency_scale": 0.8,
    #     "num_thread": 8
    # }

    return options


def get_num_ctx_by_model_name(model_name: str, default_num_ctx: int = 2048) -> int:
    # num_predict	Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)	int	num_predict 42
    # top_k	Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)	int	top_k 40
    # top_p	Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)

    # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

    # DEFAULT: 2048
    # mixtral num_ctx:64k    https://ollama.com/library/mixtral
    # llama3 num_ctx: 8k    https://ollama.com/library/llama3-gradient
    # llama3 num_ctx: 128k    https://ollama.com/library/llama3.1
    # llama3-gradient: 256000   <1M https://ollama.com/library/llama3-gradient
    num_ctx: int = default_num_ctx
    match model_name:
        case s if s.startswith("deepseek-r1"):
            # trained context length: 131_072
            num_ctx = 64 * 1024
            # num_ctx = 32 * 1024
        case s if s.startswith("bespoke-minicheck"):
            num_ctx = 32 * 1024
        case s if s.startswith("reader-lm"):
            num_ctx = 256 * 1024
            # num_ctx = 32 * 1024
        case "hermes3:70b-llama3.1-q4_0":
            num_ctx = 8_192
        case s if s.startswith("hermes3"):
            num_ctx = 64 * 1024  # 128 * 1024 -> 9%CPU
            if s == "hermes3:latest":
                num_ctx = 128 * 1024
        case s if s.find("mixtral") >= 0:
            num_ctx = 64 * 1024
            if s.find("q8_0") >= 0:
                # beim qbit-8 ist 12 das höchste, das "gut" geht...
                num_ctx = 12 * 1024  # 64*1024
        case s if s.find("gradient") >= 0:
            # num_ctx = 64 * 1024
            num_ctx = 32 * 1024
            if s.find("70b") >= 0:
                num_ctx = 12 * 1024
        case s if s.startswith("llama3.1"):
            num_ctx = 64 * 1024
        case s if s.startswith("llama3"):
            num_ctx = 8_192
        case s if s.startswith("gemma"):
            num_ctx = 8_192  # 4_096

    return num_ctx


settings: Settings = Settings()  # type: ignore


if settings.postgresql.url:
    os.environ["PSQL_DB_URL"] = os.getenv("PSQL_DB_URL", settings.postgresql.url)

os.environ["PSQL_DB_HOST"] = os.getenv("PSQL_DB_HOST", settings.postgresql.host_in_cluster if is_in_cluster() else settings.postgresql.host)  # type: ignore
os.environ["PSQL_DB_USERNAME"] = os.getenv("PSQL_DB_USERNAME", settings.postgresql.username)
os.environ["PSQL_DB_PASSWORD"] = os.getenv("PSQL_DB_PASSWORD", settings.postgresql.password)
os.environ["PSQL_DB_NAME"] = os.getenv("PSQL_DB_NAME", settings.postgresql.dbname)
os.environ["PSQL_DB_PORT"] = os.getenv("PSQL_DB_PORT", str(settings.postgresql.port))

# HOST: str = "http://localhost:11434"
# assuming HTTP -> no HTTPS atm
OLLAMA_HOST: str = os.getenv("OLLAMA_BASE_URL", f"http://{settings.ollama.host}:80")

if is_in_cluster():
    OLLAMA_HOST = f"http://{settings.ollama.host_in_cluster}:{settings.ollama.port}"


OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", settings.ollama.ollama_model)
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_MODEL", settings.ollama.ollama_embed_model)

OLLAMA_FUNCTION_CALLING_MODEL: str = os.getenv(
    "OLLAMA_FUNCTION_CALLING_MODEL", settings.ollama.ollama_function_calling_model
)
OLLAMA_GUESS_LANGUAGE_MODEL: str = os.getenv("OLLAMA_GUESS_LANGUAGE_MODEL", settings.ollama.ollama_guess_language_model)
CHROMADB_DEFAULT_COLLECTION_NAME: str = os.getenv("CHROMADB_DEFAULT_COLLECTION_NAME", settings.chromadb.default_collectionname)  # type: ignore

ARLEY_AUG_UNIFIED: bool = str2bool(os.getenv("ARLEY_AUG_UNIFIED", settings.arley_aug.unified))
ARLEY_AUG_PER_ITEM: bool = str2bool(os.getenv("ARLEY_AUG_PER_ITEM", settings.arley_aug.per_item))
ARLEY_AUG_NUM_DOCS: int = int(os.getenv("ARLEY_AUG_NUM_DOCS", settings.arley_aug.num_docs))
ARLEY_AUG_TEMPLATE_TYPE: TemplateType = TemplateType[
    os.getenv("ARLEY_AUG_TEMPLATE_TYPE", settings.arley_aug.template_type.value)
]
ARLEY_AUG_ONLY_CONTRACTS: bool = str2bool(os.getenv("ARLEY_AUG_ONLY_CONTRACTS", settings.arley_aug.only_contracts))
ARLEY_AUG_LANG_FILTER: bool = str2bool(os.getenv("ARLEY_AUG_LANG_FILTER", settings.arley_aug.lang_filter))
ARLEY_AUG_FIRST_REQUEST_INCLUDE_AUG: bool = str2bool(
    os.getenv("ARLEY_AUG_FIRST_REQUEST_INCLUDE_AUG", settings.arley_aug.first_request_include_aug)
)
ARLEY_AUG_FIRST_REQUEST_UNIFIED: bool = str2bool(
    os.getenv("ARLEY_AUG_FIRST_REQUEST_UNIFIED", settings.arley_aug.first_request_unified)
)
ARLEY_AUG_FIRST_REQUEST_PER_ITEM: bool = str2bool(
    os.getenv("ARLEY_AUG_FIRST_REQUEST_PER_ITEM", settings.arley_aug.first_request_per_item)
)
ARLEY_AUG_FIRST_REQUEST_TEMPLATE_TYPE: TemplateType = TemplateType[
    os.getenv("ARLEY_AUG_FIRST_REQUEST_TEMPLATE_TYPE", settings.arley_aug.first_request_template_type.value)
]
ARLEY_AUG_FIRST_REQUEST_N_AUG_RESULTS: int = int(
    os.getenv("ARLEY_AUG_FIRST_REQUEST_N_AUG_RESULTS", settings.arley_aug.first_request_n_aug_results)
)
ARLEY_AUG_FIRST_REQUEST_AUG_ONLY_CONTRACTS: bool = str2bool(
    os.getenv("ARLEY_AUG_FIRST_REQUEST_AUG_ONLY_CONTRACTS", settings.arley_aug.first_request_aug_only_contracts)
)
ARLEY_AUG_FIRST_REQUEST_AUG_LANG_FILTER: bool = str2bool(
    os.getenv("ARLEY_AUG_FIRST_REQUEST_AUG_LANG_FILTER", settings.arley_aug.first_request_aug_lang_filter)
)
REFINELOG_RECIPIENTS: list[str] | None = None
if os.getenv("REFINELOG_RECIPIENTS"):
    REFINELOG_RECIPIENTS = os.getenv("REFINELOG_RECIPIENTS").split(",")  # type: ignore

# TODO HT 20240917 -> move to proper pydantic-settings submodel-targeting e.g. "ARLEY_AUG_UNIFIED" -> "ARLEY__AUG_UNIFIED"

TEMPLATE_VERSION: str = os.getenv("ARLEY_AUG_TEMPLATE_VERSION", settings.arley_aug.template_version)

TEMPLATEDIRPATH: Path = Path(__file__).parent.resolve()
TEMPLATEDIRPATH = Path(TEMPLATEDIRPATH, "llm")
TEMPLATEDIRPATH = Path(TEMPLATEDIRPATH, "templates")
TEMPLATEDIRPATH = Path(TEMPLATEDIRPATH, TEMPLATE_VERSION)


# TODO HT 20250331 also put in settings-class and yaml
ARLEY_IMAPLOOP_MAX_IDLE_LOOPS: int | None = None
if os.getenv("ARLEY_IMAPLOOP_MAX_IDLE_LOOPS") is not None:
    ARLEY_IMAPLOOP_MAX_IDLE_LOOPS = int(os.getenv("ARLEY_IMAPLOOP_MAX_IDLE_LOOPS"))  # type: ignore
else:
    ARLEY_IMAPLOOP_MAX_IDLE_LOOPS = 20

ARLEY_IMAPLOOP_MAX_IDLE_UNSUCCESS_IN_SEQUENCE: int = int(
    os.getenv("ARLEY_IMAPLOOP_MAX_IDLE_UNSUCCESS_IN_SEQUENCE", "5")
)
ARLEY_IMAPLOOP_TIMEOUT_PER_IDLE_LOOP: int = int(os.getenv("ARLEY_IMAPLOOP_TIMEOUT_PER_IDLE_LOOP", "10"))
ARLEY_OLLAMALOOP_TIMEOUT_PER_LOOP: int = int(os.getenv("ARLEY_OLLAMALOOP_TIMEOUT_PER_LOOP", "10"))

ARLEY_OLLAMALOOP_MAX_LOOPS: int | None = None
if os.getenv("ARLEY_OLLAMALOOP_MAX_LOOPS") is not None:
    ARLEY_OLLAMALOOP_MAX_LOOPS = int(os.getenv("ARLEY_OLLAMALOOP_MAX_LOOPS"))  # type: ignore


logger.info(f"Effective OLLAMA_MODEL: {OLLAMA_MODEL}")
logger.info(f"Effective OLLAMA_GUESS_LANGUAGE_MODEL: {OLLAMA_GUESS_LANGUAGE_MODEL}")
logger.info(f"Effective OLLAMA_EMBED_MODEL: {OLLAMA_EMBED_MODEL}")
logger.info(f"Effective templatedirpath: {TEMPLATEDIRPATH}")


if __name__ == "__main__":
    log_settings()
