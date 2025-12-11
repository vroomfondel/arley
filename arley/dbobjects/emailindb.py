# from __future__ import annotations

import datetime
import os
import time
import uuid
from collections import defaultdict
from email import utils
from email.headerregistry import Address
from enum import StrEnum, auto
from io import StringIO
from typing import Callable, ClassVar, List, Literal, Optional, Self, get_args
from uuid import UUID

import pytz
import sqlalchemy
from loguru import logger
from sqlalchemy import Boolean, DateTime, Enum, MetaData, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, Session, mapped_column

import dbaccess.dbconnection
from arley import Helper
from arley.config import is_in_cluster, settings
from arley.emailinterface.myemailmessage import MyEmailMessage
from dbaccess.db_object import DBObject, DBObjectNEW
from dbaccess.dbconnection import (DBConnectionEngine,
                                   DBObjectInsertUpdateDeleteResult)

# from sqlalchemy import Dialect, VARCHAR, String, Executable
# from sqlalchemy.sql.type_api import TypeEngine, _T




TIMEZONE: datetime.tzinfo = pytz.timezone(settings.timezone)

# monkey-patch
DBObject.TIMEZONE = TIMEZONE


# CHECK: https://docs.sqlalchemy.org/en/20/core/compiler.html

# arley=# create type result as enum ('undefined', 'pending', 'working', 'rejected','failed','processed');
# CREATE TYPE
# arley=# \dT+ result
#                                        List of data types
#  Schema |  Name  | Internal name | Size | Elements  |  Owner   | Access privileges | Description
# --------+--------+---------------+------+-----------+----------+-------------------+-------------
#  public | result | result        | 4    | undefined+| postgres |                   |
#         |        |               |      | pending  +|          |                   |
#         |        |               |      | working  +|          |                   |
#         |        |               |      | rejected +|          |                   |
#         |        |               |      | failed   +|          |                   |
#         |        |               |      | processed |          |                   |
# (1 row)

# arley=> create table emails (emailid UUID not null, received timestamp with time zone not null default now(), processed bool default false, processresult result not null default 'undefined', rootemailid UUID, isrootemail bool default false, frommail varchar(255), tomail varchar(255), toarley bool default false, fromarley bool default false, sequencenumber int not null default 0, envelopeemailid varchar(255), subject text, mailbody text, ollamaresponse jsonb, lang character(2) default 'de', ollamamsgs jsonb '{}'::jsonb, primary key(emailid));
#
# arley=> \d emails
#                                      Tabelle »public.emails«
#      Spalte      |           Typ            | Sortierfolge | NULL erlaubt? |     Vorgabewert
# -----------------+--------------------------+--------------+---------------+---------------------
#  emailid         | uuid                     |              | not null      |
#  received        | timestamp with time zone |              | not null      | now()
#  processed       | boolean                  |              |               | false
#  processresult   | result                   |              | not null      | 'undefined'::result
#  rootemailid     | uuid                     |              |               |
#  isrootemail     | boolean                  |              |               | false
#  frommail        | character varying(255)   |              |               |
#  tomail          | character varying(255)   |              |               |
#  toarley         | boolean                  |              |               | false
#  fromarley       | boolean                  |              |               | false
#  sequencenumber  | integer                  |              | not null      | 0
#  envelopeemailid | character varying(255)   |              |               |
#  subject         | text                     |              |               |
#  mailbody        | text                     |              |               |
#  ollamaresponse  | jsonb                    |              |               |
#  lang            | character(2)             |              |               | 'de'::bpchar
#  ollamamsgs      | jsonb                    |              |               | '{}'::jsonb
# Indexe:
#     "emails_pkey" PRIMARY KEY, btree (emailid)


class Result(StrEnum):
    undefined = auto()
    pending = auto()
    working = auto()
    rejected = auto()
    failed = auto()
    processed = auto()

    # def __repr__(self):
    #     return self.name


import sqlalchemy.engine.default
# from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.dialects import postgresql

fart = sqlalchemy.engine.default.DefaultExecutionContext.get_result_processor


# def furt(self, type_, colname, coltype):
#     logger.debug(f"{colname=}\t{type(type_)=} {type_=}\t{type(coltype)=} {coltype=}")
#     fart(self, type_, colname, coltype)
#
#
# sqlalchemy.engine.default.DefaultExecutionContext.get_result_processor = furt
#
#
# import sqlalchemy.engine.cursor
#
# fmeta = sqlalchemy.engine.cursor.CursorResultMetaData._merge_cursor_description
#
#
# def fmetaurt(
#     self,
#     context,
#     cursor_description,
#     result_columns,
#     num_ctx_cols,
#     cols_are_ordered,
#     textual_ordered,
#     ad_hoc_textual,
#     loose_column_name_matching,
# ):
#
#     # type(context)=<class 'sqlalchemy.dialects.postgresql.psycopg2.PGExecutionContext_psycopg2'> context=<sqlalchemy.dialects.postgresql.psycopg2.PGExecutionContext_psycopg2 object at 0x7fcdafea45c0>
#     # type(result_columns)=<class 'list'> result_columns=[]
#     logger.debug(f"{type(context)=} {context=}\t{type(result_columns)=} {result_columns=}")
#     return fmeta(
#         self,
#         context,
#         cursor_description,
#         result_columns,
#         num_ctx_cols,
#         cols_are_ordered,
#         textual_ordered,
#         ad_hoc_textual,
#         loose_column_name_matching,
#     )
#
#
# sqlalchemy.engine.cursor.CursorResultMetaData._merge_cursor_description = fmetaurt

# postgresql.dialect().getresult_processor
# sqlalchemy.dialects.postgresql.base.PGCompiler.render_literal_value = fufu
# sqlalchemy.dialects.postgresql.base.PGTypeCompiler.render_literal_value = fufut
#
# logger.debug(f"{postgresql.dialect().type_compiler=}")
# logger.debug(f"{postgresql.dialect().type_compiler_cls=}")
# logger.debug(f"{postgresql.dialect().type_compiler_instance=}")
#
# logger.debug(f"{postgresql.dialect().statement_compiler.render_literal_value=}")

# @compiles(String)
# @compiles(VARCHAR)
# def compile_varchar(element, compiler, **kw):
#     logger.debug("COMPILE VARCHAR")
#     if element. length == 'max':
#         return "VARCHAR('max')"
#     else:
#         return compiler. visit_VARCHAR(element, **kw)
#
#
# class SelectStuff(Executable):
#     inherit_cache = False
#     def __init__(self, table, select):
#         self.table = table
#         self.select = select
#
# @compiles(SelectStuff)
# def visit_select(element, compiler, **kw):
#     print("SELECT VISITED")
#     return None

# @compiles(Result)
# class MyType(types.TypeDecorator):
#     '''Prefixes Unicode values with "PREFIX:" on the way in and
#     strips it off on the way out.
#     '''
#
#     impl = TypeEngine
#
#     def process_bind_param(self, value, dialect):
#         return "PREFIX:" + value
#
#     def process_result_value(self, value, dialect):
#         return value[7:]
#
#     def copy(self, **kw):
#         return MyType(self.impl.length)
# # The class-level “impl” attribute is required, and can reference any TypeEngine class. Alternatively, the load_dialect_impl() method can be used to provide different type classes based on the dialect given; in this case, the “impl” variable can reference TypeEngine as a placeholder.
# #
# # Types that receive a Python type that isn’t similar to the ultimate type used may want to define the TypeDecorator.coerce_compared_value() method. This is used to give the expression system a hint when coercing Python objects into bind parameters within expressions. Consider this expression:
# #
#
#
# class _EnhDateTime(datetime.datetime):
#
#     def foo(self):
#         return 'foo'
#
#
# class EnhDateTime(TypeDecorator):
#     impl = UUID
#     def process_literal_param(
#         self, value: Optional[_T], dialect: Dialect
#     ) -> str:
#         print("wonk")
#         return None
#
#     def bind_processor(self, dialect: Dialect) -> Callable:
#         print("WORKUS")
#         return None
#
#     def process_result_value(self, value, dialect):
#         print("WOOHOO")
#         if value is not None:
#             value = _EnhDateTime(
#                 value.year, value.month, value.day, value.hour, value.minute,
#                 value.second, value.microsecond, value.tzinfo
#             )
#         return value
#
#
#
# class ResultType(types.TypeDecorator):
#     logger = logger.bind(classname=__qualname__)
#
#     impl = types.VARCHAR  #types.TypeEngine  # 4  # TypeEngine
#
#     cache_ok = False
#
#     def load_dialect_impl(self, dialect):
#         print("WOOHOO1")
#
#         if dialect.name == "postgresql":
#             return dialect.type_descriptor(UUID())
#         else:
#             return dialect.type_descriptor(self._default_type)
#
#
#     def process_bind_param(self, value, dialect):
#         print("WOOHOO2")
#         ResultType.logger.debug(f"{dialect=} {value=}")
#         return super().process_bind_param(value, dialect)
#
#     def process_result_value(self, value, dialect):
#         print("WOOHOO3")
#         ResultType.logger.debug(f"{dialect=} {value=}")
#         return super().process_result_value(value, dialect)
#
#     def copy(self, **kw):
#         print("WOOHOO4")
#         return ResultType(self.impl.length)


class ArleyEmailInDBNEW(DBObjectNEW):
    logger: ClassVar = logger.bind(classname=f"{__qualname__}")

    __tablename__ = "emails"

    emailid: Mapped[UUID] = mapped_column(primary_key=True)
    received: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    processed: Mapped[bool] = mapped_column(Boolean, server_default="t", default=False)
    processresult: Mapped[Result] = mapped_column(
        Enum(Result, create_constraint=True, validate_strings=True, name="result"),
        server_default=text("'undefined'::result"),
        default=Result.undefined.value,
    )
    rootemailid: Mapped[UUID] = mapped_column(sqlalchemy.UUID)
    isrootemailid: Mapped[bool] = mapped_column(Boolean, server_default="t", default=False)
    frommail: Mapped[str] = mapped_column(sqlalchemy.String(255))
    tomail: Mapped[str] = mapped_column(sqlalchemy.String(255))
    toarley: Mapped[bool] = mapped_column(Boolean, server_default="t", default=False)
    fromarley: Mapped[bool] = mapped_column(Boolean, server_default="t", default=False)
    sequencenumber: Mapped[int] = mapped_column(sqlalchemy.Integer, server_default="0", default=0, nullable=False)
    envelopeemailid: Mapped[str] = mapped_column(sqlalchemy.String(255))
    subject: Mapped[str] = mapped_column(sqlalchemy.Text)
    mailbody: Mapped[str] = mapped_column(sqlalchemy.Text)
    ollamaresponse: Mapped[dict] = mapped_column(JSONB)
    lang: Mapped[Literal["de", "en"]] = mapped_column(sqlalchemy.CHAR(2), default="de", server_default="de")
    ollamamsgs: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))

    def sanitize_header_fields(self) -> None:
        if self.envelopeemailid:
            self.envelopeemailid = self.envelopeemailid.replace("\n", "").replace("\r", "").strip()

        if self.frommail:
            self.frommail = self.frommail.replace("\n", "").replace("\r", "")

        if self.tomail:
            self.tomail = self.tomail.replace("\n", "").replace("\r", "")

        if self.subject:
            self.subject = self.subject.replace("\n", "").replace("\r", "")

    @classmethod
    def insert_from_myemail(
        cls,
        myemail: MyEmailMessage,
        rootemailid: UUID | None = None,
        arley_email: str = settings.emailsettings.mailaddress,
        lang: Literal["de", "en"] = "de",
    ) -> "ArleyEmailInDBNEW | None":
        # TODO HT 20240920 REWRITE TO SQLALCHEMY LOGIC!!!

        sequencenumber: int = 0
        if rootemailid is not None:
            sql: str = f"select count(*) as countme from emails where rootemailid='{rootemailid}'"
            cls.logger.debug(f"{sql=}")
            rs: dict | None = dbaccess.dbconnection.exec_one_line(sql)
            cls.logger.debug(rs)
            if rs is not None:
                sequencenumber = rs["countme"]

        aeid: ArleyEmailInDBNEW = ArleyEmailInDBNEW()
        aeid.emailid = uuid.uuid4()
        aeid.lang = lang
        aeid.processed = False
        aeid.processresult = Result.pending

        received: datetime.datetime | None = myemail.get_date_from_last_received_header()
        if received is None:
            received = myemail.get_date_from_date_header()

        if received is not None:
            aeid.received = received

        if not aeid.received:
            aeid.received = datetime.datetime.now(tz=DBObject.TIMEZONE)

        aeid.rootemailid = rootemailid if rootemailid else aeid.emailid  # self-root-email...
        aeid.isrootemailid = False if rootemailid else True

        aeid.frommail = myemail.from_email
        aeid.tomail = myemail.to_email

        aeid.subject = myemail.subject
        aeid.toarley = aeid.tomail == arley_email
        cls.logger.debug(f"{aeid.toarley=} {aeid.tomail=} == {arley_email=}")
        aeid.fromarley = aeid.frommail == arley_email

        aeid.sequencenumber = sequencenumber + 1 if rootemailid else 0

        aeid.envelopeemailid = (
            myemail.envelope_message_id.strip()
        )  # utils.make_msgid(domain=Address(addr_spec=aeid.frommail).domain)

        mb: str | None = myemail.get_latest_reply()
        if mb is not None:
            aeid.mailbody = mb
        aeid.ollamamsgs = {}
        aeid.ollamaresponse = {}

        aeid.sanitize_header_fields()

        # TODO HT 20240920 REWRITE TO SQLALCHEMY LOGIC!!!
        if hasattr(aeid, "insertnew"):
            res: DBObjectInsertUpdateDeleteResult | None = aeid.insertnew()
            if res is not None:
                logger.debug(f"{type(res)=}\t{res=} {res.exception_occured()=} {res.get_rows_affected()=}")

            if res is not None and res.exception_occured():
                exme: Exception | None = res.get_exception()
                if exme is not None:
                    logger.error(Helper.get_exception_tb_as_string(exme))
                return None

            # rs: ArleyEmailInDB = ArleyEmailInDB.get_one_from_sql("select * from emails limit 1")
            # logger.debug(f"{type(rs)=} {rs=}")
            # logger.debug(Helper.get_pretty_dict_json_no_sort(rs.repr_json()))

        return aeid


class ArleyEmailInDB(DBObject):
    logger = logger.bind(classname=__qualname__)
    dbtablename: str = "emails"
    dbcolnames: List[str] = [
        "emailid",
        "received",
        "processed",
        "processresult",
        "rootemailid",
        "isrootemail",
        "frommail",
        "tomail",
        "toarley",
        "fromarley",
        "sequencenumber",
        "envelopeemailid",
        "subject",
        "mailbody",
        "ollamaresponse",
        "lang",
        "ollamamsgs",
    ]  # all lowercase - otherwise will need to quote!!!

    def __init__(self) -> None:
        self.emailid: Optional[UUID] = None
        self.received: Optional[datetime.datetime] = None
        self.processed: Optional[bool] = None
        self.processresult: Optional[Result] = None
        self.rootemailid: Optional[UUID] = None
        self.isrootemail: Optional[bool] = None
        self.frommail: Optional[str] = None
        self.tomail: Optional[str] = None
        self.toarley: Optional[bool] = None
        self.fromarley: Optional[bool] = None
        self.sequencenumber: Optional[int] = None
        self.envelopeemailid: Optional[str] = None
        self.subject: Optional[str] = None
        self.mailbody: Optional[str] = None
        self.ollamaresponse: Optional[dict] = None
        self.ollamamsgs: Optional[list] = None
        self.lang: Optional[Literal["de", "en"]] = None

        self.sanitize_header_fields()

        super(ArleyEmailInDB, self).__init__()

    def sanitize_header_fields(self) -> None:
        if self.envelopeemailid:
            self.envelopeemailid = self.envelopeemailid.replace("\n", "").replace("\r", "").strip()

        if self.frommail:
            self.frommail = self.frommail.replace("\n", "").replace("\r", "")

        if self.tomail:
            self.tomail = self.tomail.replace("\n", "").replace("\r", "")

        if self.subject:
            self.subject = self.subject.replace("\n", "").replace("\r", "")

    @classmethod
    def insert_from_myemail(
        cls,
        myemail: MyEmailMessage,
        rootemailid: UUID | None = None,
        arley_email: str = settings.emailsettings.mailaddress,
        lang: Literal["de", "en"] = "de",
    ) -> "ArleyEmailInDB | None":
        sequencenumber: int = 0
        if rootemailid is not None:
            sql: str = f"select count(*) as countme from emails where rootemailid='{rootemailid}'"
            cls.logger.debug(f"{sql=}")
            rs: dict | None = dbaccess.dbconnection.exec_one_line(sql)
            cls.logger.debug(rs)
            if rs is not None:
                sequencenumber = rs["countme"]

        aeid: ArleyEmailInDB = ArleyEmailInDB()
        aeid.emailid = uuid.uuid4()
        aeid.lang = lang
        aeid.processed = False
        aeid.processresult = Result.pending

        aeid.received = myemail.get_date_from_last_received_header()
        if not aeid.received:
            aeid.received = myemail.get_date_from_date_header()

        if not aeid.received:
            aeid.received = datetime.datetime.now(tz=DBObject.TIMEZONE)

        aeid.rootemailid = rootemailid if rootemailid else aeid.emailid  # self-root-email...
        aeid.isrootemail = False if rootemailid else True

        aeid.frommail = myemail.from_email
        aeid.tomail = myemail.to_email

        aeid.subject = myemail.subject
        aeid.toarley = aeid.tomail == arley_email
        cls.logger.debug(f"{aeid.toarley=} {aeid.tomail=} == {arley_email=}")
        aeid.fromarley = aeid.frommail == arley_email

        aeid.sequencenumber = sequencenumber + 1 if rootemailid else 0

        aeid.envelopeemailid = (
            myemail.envelope_message_id.strip()
        )  # utils.make_msgid(domain=Address(addr_spec=aeid.frommail).domain)

        aeid.mailbody = myemail.get_latest_reply()
        aeid.ollamamsgs = list()
        aeid.ollamaresponse = {}

        aeid.sanitize_header_fields()

        res: DBObjectInsertUpdateDeleteResult | None = aeid.insertnew()
        if res is not None:
            logger.debug(f"{type(res)=}\t{res=} {res.exception_occured()=} {res.get_rows_affected()=}")

            if res.exception_occured():
                logger.error(Helper.get_exception_tb_as_string(res.get_exception()))
                return None

        # rs: ArleyEmailInDB = ArleyEmailInDB.get_one_from_sql("select * from emails limit 1")
        # logger.debug(f"{type(rs)=} {rs=}")
        # logger.debug(Helper.get_pretty_dict_json_no_sort(rs.repr_json()))

        return aeid


# arley=# create table rawemails (emailid UUID not null, rawemail text, textmailbody text, primary key(emailid));
#
# arley=# \d rawemails
#              Table "public.rawemails"
#   Column      | Type | Collation | Nullable | Default
# --------------+------+-----------+----------+---------
#  emailid      | uuid |           | not null |
#  rawemail     | text |           |          |
#  textmailbody | text |           |          |
# Indexes:
#     "rawemails_pkey" PRIMARY KEY, btree (emailid)


class ArleyRawEmailInDB(DBObject):
    logger = logger.bind(classname=__qualname__)
    dbtablename: str = "rawemails"
    dbcolnames: List[str] = ["emailid", "rawemail", "textmailbody"]  # all lowercase - otherwise will need to quote!!!

    def __init__(self) -> None:
        self.emailid: Optional[UUID] = None
        self.rawemail: Optional[str] = None
        self.textmailbody: Optional[str] = None

        super().__init__()

    @classmethod
    def insert_rawemail_from_myemail(
        cls, myemail: MyEmailMessage, arleyemailindb: ArleyEmailInDB
    ) -> "ArleyRawEmailInDB | None":
        aeid: ArleyRawEmailInDB = ArleyRawEmailInDB()
        aeid.emailid = arleyemailindb.emailid
        aeid.rawemail = myemail.email_message.as_string()

        aeid.textmailbody = myemail.get_textbody()

        res: DBObjectInsertUpdateDeleteResult | None = aeid.insertnew()
        if res is not None:
            logger.debug(f"{type(res)=}\t{res=} {res.exception_occured()=} {res.get_rows_affected()=}")

        if res is not None and res.exception_occured():
            logger.error(Helper.get_exception_tb_as_string(res.get_exception()))
            return None

        return aeid

    # def fromRowDict(self, row: dict) -> Feedbacks:
    #         ret: Feedbacks = super().fromRowDict(row)
    #         if ret is not None:
    #             self.status = FeedbackStatus._value2member_map_[self.status]
    #             return self
    #
    #         return None
    #
    #     def fromResultSetNextRow(self, rs: sqlalchemy.engine.cursor.Result, colnames: list = None) -> Feedbacks:
    #         ret: Feedbacks = super().fromResultSetNextRow(rs, colnames)
    #         if ret is not None:
    #             self.status = FeedbackStatus._value2member_map_[self.status]
    #             return self
    #
    #         return None


# sqlalchemy.sql.sqltypes.Enum
# sqlalchemy.dialects.postgresql.named_types.ENUM


def run_new() -> None:
    dburl: str = sqlalchemy.engine.url.URL.create(
        drivername="postgresql+psycopg2",
        username=settings.postgresql.username,
        password=settings.postgresql.password,
        database=settings.postgresql.dbname,
        host=settings.postgresql.host_in_cluster if is_in_cluster() else settings.postgresql.host,
        port=settings.postgresql.port,
        # query={"host": unix_socket_path},  # if going via socket.
    ).render_as_string()
    os.environ["PSQL_DB_URL"] = dburl

    # DBObjectNEW.metadata.reflect(bind=DBConnectionEngine.get_instance().get_engine(autocommit=False))
    # DBObjectNEW.metadata.create_all(DBConnectionEngine.get_instance().get_engine())

    with Session(bind=DBConnectionEngine.get_instance().get_engine(autocommit=False)) as session:
        stmt = sqlalchemy.select(ArleyEmailInDBNEW).where(ArleyEmailInDBNEW.fromarley.is_(True))

        for emailindb in session.scalars(stmt):
            print(emailindb)

    # with Session(engine) as session:
    # ...     spongebob = User(
    # ...         name="spongebob",
    # ...         fullname="Spongebob Squarepants",
    # ...         addresses=[Address(email_address="spongebob@sqlalchemy.org")],
    # ...     )
    # ...     sandy = User(
    # ...         name="sandy",
    # ...         fullname="Sandy Cheeks",
    # ...         addresses=[
    # ...             Address(email_address="sandy@sqlalchemy.org"),
    # ...             Address(email_address="sandy@squirrelpower.org"),
    # ...         ],
    # ...     )
    # ...     patrick = User(name="patrick", fullname="Patrick Star")
    # ...
    # ...     session.add_all([spongebob, sandy, patrick])
    # ...
    # ...     session.commit()


def run_old() -> None:

    r = ArleyEmailInDB.get_one_from_sql(f"select * from emails where emailid='842137ab-fb46-4b9f-8236-f8e03226d32e'")
    if r is not None:
        logger.debug(Helper.get_pretty_dict_json_no_sort(r.repr_json()))

    exit(0)

    # undef: Result = Result.failed
    # print(f"{type(undef)=}\t{undef=}")
    #
    # testdata: ArleyEmailInDB = test_insert()
    #
    # time.sleep(10)
    # testdata.processresult = Result.working
    # testdata.save()


if __name__ == "__main__":
    run_new()

    exit(0)
