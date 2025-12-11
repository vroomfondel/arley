import datetime
import hashlib
import json
import re
import sys
import traceback
import uuid
# import functools
# https://docs.python.org/3/library/functools.html
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, List, Literal, Optional,
                    Self, Tuple, TypeVar)
from uuid import UUID

import pytz
import ruamel.yaml
# from redis import Redis #StrictRedis #since 3.0 StrictRedis IS Redis
from redis import ConnectionPool, Redis
from reputils import MailReport

try:
    from .config import settings
except ImportError:
    from arley.config import settings

from loguru import logger

# import redis.commands.search.aggregation as aggregations
# import redis.commands.search.reducers as reducers
# from redis.commands.search.field import TextField, NumericField, TagField
# from redis.commands.search.indexDefinition import IndexDefinition, IndexType
# from redis.commands.search.query import NumericFilter, Query

T = TypeVar("T")


# https://stackoverflow.com/questions/6760685/what-is-the-best-way-of-implementing-singleton-in-python
class Singleton(type):
    # logger = logger.bind(classname=__qualname__)
    _instances: ClassVar[Dict[type, Any]] = {}

    def __call__(cls: type[T], *args: Any, **kwargs: Any) -> T:
        # cls.logger.debug(f"{args=}")
        # cls.logger.debug(f"{kwargs=}")

        if cls not in cls._instances:  # type: ignore
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)  # type: ignore

        return cls._instances[cls]  # type: ignore


redisconnectionpool: ConnectionPool = ConnectionPool(
    host=settings.redis.host,
    port=settings.redis.port,
    db=0,
    decode_responses=True,
    protocol=3,
)
# >>> r = redis.Redis(connection_pool=pool)


def get_redis() -> Redis:
    return Redis(connection_pool=redisconnectionpool)
    # redis: Redis = Redis(
    #     host=os.getenv("REDIS_HOST"),
    #     port=int(os.getenv("REDIS_PORT")),
    #     db=0,
    #     decode_responses=True,
    #     protocol=3,
    #   )


def get_exception_tb_as_string(exc: Exception | None) -> str | None:
    if exc is None:
        return None

    tb1: traceback.TracebackException = traceback.TracebackException.from_exception(exc)
    tbsG = tb1.format()
    tbs = ""

    for line in tbsG:
        tbs = tbs + "\n" + line

    return tbs


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if hasattr(obj, "repr_json"):
            return obj.repr_json()
        elif hasattr(obj, "as_string"):
            return obj.as_string()
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()  # strftime("%Y-%m-%d %H:%M:%S %Z")
        elif isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, datetime.timedelta):
            return str(obj)
        elif isinstance(obj, dict) or isinstance(obj, list):
            robj: str = get_pretty_dict_json_no_sort(obj)
            return robj
        else:
            return json.JSONEncoder.default(self, obj)


def print_pretty_dict_json(data: Any, indent: int = 4) -> None:
    print(json.dumps(data, indent=indent, sort_keys=True, cls=ComplexEncoder, default=str))


def get_pretty_dict_json(data: Any, indent: int = 4) -> str:
    return json.dumps(data, indent=indent, sort_keys=True, cls=ComplexEncoder, default=str)


def get_pretty_dict_json_no_sort(data: Any, indent: int = 4) -> str:
    return json.dumps(data, indent=indent, sort_keys=False, cls=ComplexEncoder, default=str)


def set_redis_str(
    name: str, value: str, ex: Optional[int] = None, nx: bool = False, get: bool = False
) -> Optional[str | bool]:
    ret: Optional[str | bool | Literal[""]] = get_redis().set(name=name, value=value, ex=ex, nx=nx, get=get)  # type: ignore
    return ret


def get_redis_str(name: str) -> Optional[str]:
    ret: Optional[str] = get_redis().get(name)  # type: ignore
    return ret


def drop_redis_entry(name: str) -> None:
    get_redis().delete(name)


def drop_redis_entries(*names: str) -> None:
    for name in names:
        # print(f"NAME: {name}")
        get_redis().delete(name)


def set_redis_json(
    name: str,
    value: Dict,
    ex: Optional[int] = None,
    nx: bool = False,
    get: bool = False,
) -> Optional[Dict | bool | Literal[""]]:
    ret_1: Optional[str | bool] = set_redis_str(name=name, value=json.dumps(value, default=str), ex=ex, nx=nx, get=get)
    if ret_1 and isinstance(ret_1, str):
        return json.loads(ret_1)

    return ret_1


def get_redis_json(name: str) -> Optional[Dict]:
    ret_1: Optional[str] = get_redis().get(name)  # type: ignore
    if ret_1:
        return json.loads(ret_1)
    return None


def rediscached(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that caches the results of the function call.

    We use Redis in this example, but any cache (e.g. memcached) will work.
    We also assume that the result of the function can be seralized as JSON,
    which obviously will be untrue in many situations. Tweak as needed.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Generate the cache key from the function's arguments.
        key_parts = [func.__name__] + list(args)
        key = "-".join(key_parts)
        result = get_redis().get(key)

        if result is None:
            # Run the function and cache the result for next time.
            value = func(*args, **kwargs)
            value_json = json.dumps(value)
            get_redis().set(key, value_json)
        else:
            # Skip the function entirely and use the cached value instead.
            value = json.loads(result)  # type: ignore

        return value

    return wrapper


# def exception_reporter_wrapped(orig_function=None, *, re_raise_error: bool = True):
#     def decorator_exception_reporter_wrapped(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             try:
#                 return func(*args, **kwargs)
#             except Exception as ex:
#                 send_telegram_msg(f"<pre>{get_exception_tb_as_string(ex)}</pre>", htmlmode=True)
#                 if re_raise_error:
#                     raise ex
#
#         return wrapper
#
#     if orig_function:
#         return decorator_exception_reporter_wrapped(orig_function)
#     else:
#         return decorator_exception_reporter_wrapped


def get_md5_for_file(file: Path) -> str:
    with open(file, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(4_096):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def get_dict_as_yaml_str(data: dict) -> str:
    yaml = ruamel.yaml.YAML()
    yaml.indent(sequence=4, offset=2)
    strwriter: StringIO = StringIO()
    yaml.dump(data, stream=strwriter)
    return strwriter.getvalue()


def flatten(d: dict, parent_key: str = "") -> dict:
    items: List = []
    for k, v in d.items():
        try:
            items.extend(flatten(v, "%s%s_" % (parent_key, k)).items())
        except AttributeError:
            items.append(("%s%s" % (parent_key, k), v))
    return dict(items)


def flatten_lists(d: dict) -> dict:
    ret: dict = {}
    for k, v in d.items():
        if isinstance(v, list):
            vl: StringIO = StringIO()
            for ind, item in enumerate(v, start=1):
                if ind > 1:
                    vl.write("\n")
                vl.write(f"{ind}. {item}")
                # ret[f"{k}_{ind}"] = item
            ret[k] = vl.getvalue()
        else:
            ret[k] = v

    return ret


def detach_NOTE_line(txt: str) -> Tuple[str, str | None]:
    ret_note: Optional[str] = None
    ret: StringIO = StringIO()
    txt_reader: StringIO = StringIO(txt)

    line: str | None
    while True:
        line = txt_reader.readline()
        if not line:
            break

        if 10 > line.upper().find("NOTE:") >= 0:
            ret_note = line.strip()
        else:
            ret.write(line)

    return ret.getvalue(), ret_note


def maillog(
    subject: str, from_mail: str, text: str, mailrecipients_to: list[str], mailrecipients_cc: list[str] | None = None
) -> str:
    _timezone: datetime.tzinfo = pytz.timezone(settings.timezone)

    _sdfD_formatstring: str = "%d.%m.%Y"
    _sdfDHM_formatstring: str = "%d.%m.%Y %H:%M"
    _sdfE_formatstring: str = "%Y%m%d"

    serverinfo = MailReport.SMTPServerInfo(
        smtp_server=settings.emailsettings.smtpserver,
        smtp_port=settings.emailsettings.smtpport,
        smtp_user=settings.emailsettings.mailuser,
        smtp_pass=settings.emailsettings.mailpassword,
        useStartTLS=True,
        wantsdebug=False,
        ignoresslerrors=True,
    )

    now: datetime.datetime = datetime.datetime.now(_timezone)
    sdd: str = now.strftime(_sdfD_formatstring)

    sendmail: MailReport.MRSendmail = MailReport.MRSendmail(
        serverinfo=serverinfo,
        returnpath=MailReport.EmailAddress.fromSTR(from_mail),
        replyto=MailReport.EmailAddress.fromSTR(from_mail),
        subject=subject,
    )
    sendmail.tos = [MailReport.EmailAddress.fromSTR(k) for k in [mailrecipients_to]]

    if mailrecipients_cc is not None:
        sendmail.ccs = [MailReport.EmailAddress.fromSTR(k) for k in mailrecipients_cc]

    sent_mail: str = sendmail.send(txt=text)
    return sent_mail


_think_tag_pattern: re.Pattern[str] = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.MULTILINE | re.DOTALL)


def detach_think_tag(input_text: Optional[str]) -> Tuple[str | None, str | None]:
    if not input_text:
        return None, None

    matches: list[str | tuple[str, ...]] | None = _think_tag_pattern.findall(input_text)
    if matches:
        # logger.debug(f"#matches: {len(matches)}")

        for ind, match in enumerate(matches, start=1):
            # Extract matching values of all groups
            # logger.debug(f"match#{ind}  {type(match)=} {match=}")

            if isinstance(match, tuple):
                # logger.debug(f"{match[0]=}")
                # logger.debug(f"{match[1]=}")

                return match[1], match[0]
            else:
                # str-type
                # DOES NOT MAKE SENSE!!!
                # logger.debug(f"{match=}")
                return match, None

    # no match
    return input_text, None
