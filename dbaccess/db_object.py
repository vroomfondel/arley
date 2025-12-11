import json
from enum import StrEnum
from io import StringIO

import pytz
from sqlalchemy.ext.declarative import declarative_base

from dbaccess.dbconnection import (
    DBObjectInsertUpdateDeleteResult,
    MultiLineResSet,
    get_compiled_sql,
    DBConnectionEngine,
)

import datetime
import uuid
from typing import Optional, List, Self, Any, Dict, ClassVar

import sqlalchemy
from sqlalchemy import text, CursorResult, Connection, MetaData
from sqlalchemy.schema import Table
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship



# This object shall be any form of Object-Persistence-Baseclass trimmed for efficiency...
# "Alternatively": https://realpython.com/python-sqlite-sqlalchemy/#working-with-sqlalchemy-and-python-objects

from loguru import logger

LOGME_VERBOSE: bool = True

# Create and engine and get the metadata
# _base = declarative_base()
#
# _metadata = MetaData()
# _metadata.reflect(DBConnectionEngine.get_instance().get_engine())
# # https://stackoverflow.com/questions/6290162/how-to-automatically-reflect-database-to-sqlalchemy-declarative
#
# class Emails(_base):
#     __table__ = Table('emails', _metadata, autoload_with=DBConnectionEngine.get_instance().get_engine())
#
# class RawEmails(_base):
#     __table__ = Table('rawemails', _metadata, autoload_with=DBConnectionEngine.get_instance().get_engine())
#
#
# logger.debug(f"{_metadata=}")
# logger.debug(f"{Emails.__table__=}")
# logger.debug(f"{RawEmails.__table__=}")

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


def get_pretty_dict_json_no_sort(data: Any, indent: int = 4) -> str:
    return json.dumps(data, indent=indent, sort_keys=False, cls=ComplexEncoder, default=str)


class DBObjectNEW(DeclarativeBase):
    pass


class DBObject:
    logger: ClassVar = logger.bind(classname=f"{__qualname__}")
    TIMEZONE: datetime.tzinfo = pytz.timezone("Europe/Berlin")  # may be overriden/changed before use...

    def __init__(self) -> None:
        self.dbtablename: str|None = None
        self.dbcolnames: List[str] | None = None

    @staticmethod
    def psql_quote_str(input: str) -> str:
        if input is None or input.find("'") < 0:
            return input
        return input.replace("'", "''")  # double-quotation is psql-style - aight ?!



    def __repr__(self) -> str:
        return f"I am a {self.__class__.__qualname__}. ({self.__dir__()}"

    def repr_json(self) -> Dict:
        ret: dict = {}
        for k in self.__dict__:
            v = self.__dict__[k]

            # self.logger.debug(f"v(class={type(k)} -> {type(v)}): {k} -> {v}")

            if isinstance(v, uuid.UUID):
                ret[k] = str(v)
            elif isinstance(v, datetime.date) or isinstance(v, datetime.datetime):
                ret[k] = str(v)
            elif isinstance(v, dict) or isinstance(v, list):
                ret[k] = get_pretty_dict_json_no_sort(v)
            else:
                ret[k] = v

        return ret

    # def to_json(self):
    #     return json.dumps(self, default=lambda o: o.__dict__,
    #                       sort_keys=True, indent=4)

    def from_result_set_next_row(self, rs: sqlalchemy.engine.cursor.Result, colnames: List|None = None) -> Self|None:
        """
        Returns the object with the attributes set to the "contents" of the ResultSet
            Parameters:
                rs (sqlalchemy.engine.cursor.Result): The cursor holding the data to be set onto the object
                colnames (list[str]): A list of the columnnames in proper order.
                    If the colnames are omitted, these are fetched from the ResultSet

        """
        row = rs.fetchone()
        # print(f"colnames(type={type(colnames)}: {colnames}")
        # print(f"row(type={type(row)}: {row}")

        if row is None:
            return None

        if colnames is None:
            keys_rs: sqlalchemy.engine.result.RMKeyView = rs.keys()
            # print(keys_rs)
            # print(type(keys_rs))  # <class 'sqlalchemy.engine.result.RMKeyView'>
            colnames = list(keys_rs)

        # print(f"ROW: {row}")
        for ind, c in enumerate(row):
            setattr(self, colnames[ind], c)
            # print(f"\t[{ind} type={type(c)} name={colnames[ind]}]: {c}")

        return self

    def from_row_dict(self, row: dict) -> Self:
        if row is None:
            return None

        for key in row.keys():
            setattr(self, key, row[key])
            # LoggingHelper.debug(f"{key=} {row[key]=} {type(row[key])=}")
            # print(f"\t[{ind} type={type(c)} name={colnames[ind]}]: {c}")

        return self

    def save(self) -> DBObjectInsertUpdateDeleteResult|None:
        savesql: StringIO = StringIO()
        savesql.write(f"update {self.dbtablename} set ")

        cndict: dict = {}

        assert self.dbcolnames
        for i, cn in enumerate(self.dbcolnames):
            if i > 0:
                savesql.write(", ")

            savesql.write(f"{cn}=:{cn}")
            cndict[cn] = getattr(self, cn)

            if isinstance(cndict[cn], dict) or isinstance(cndict[cn], list):
                cndict[cn] = get_pretty_dict_json_no_sort(cndict[cn])

        savesql.write(f" where {self.dbcolnames[0]}=:{self.dbcolnames[0]}")

        if LOGME_VERBOSE:
            self.logger.debug(f"{savesql.getvalue()=} {cndict=}")

        result: DBObjectInsertUpdateDeleteResult|None = self.insertupdatedelete(
            savesql.getvalue(),
            **cndict
        )

        return result

    @classmethod
    def insertupdatedelete(cls, sqlstr: str, **kwargs: Any) -> Optional[DBObjectInsertUpdateDeleteResult]:
        # https://chartio.com/resources/tutorials/how-to-execute-raw-sql-in-sqlalchemy/
        # https://stackoverflow.com/questions/36524293/inserting-into-postgres-database-from-sqlalchemy

        conn: sqlalchemy.engine.base.Connection
        rs: Optional[sqlalchemy.engine.cursor.CursorResult] = None
        ex: Optional[Exception] = None
        with DBConnectionEngine.get_instance().connect() as conn:
            try:
                sql = text(sqlstr)

                # DBObject.logger.debug(f"{get_compiled_sql(sql.text, **kwargs)=}")

                rs = conn.execute(sql, kwargs)
                # DBObject.logger.debug(f"{type(rs)=}")
                # DBObject.logger.debug(f"{rs.rowcount=}")
                # DBObject.logger.debug(f"{rs=}")
            except Exception as exX:
                ex = exX
                # raise exX

            assert rs
            ret: DBObjectInsertUpdateDeleteResult|None = DBObjectInsertUpdateDeleteResult().from_result_set(rs, ex)

            return ret

    @classmethod
    def get_result_set_to_list(cls, rs: sqlalchemy.engine.cursor.Result) -> List[Self]:
        """
        Returns the object with the attributes set to the "contents" of the ResultSet
            Parameters:
                rs (sqlalchemy.engine.cursor.Result): The cursor holding the data to be set onto the object
                colnames (list[str]): A list of the columnnames in proper order.
                    If the colnames are omitted, these are fetched from the ResultSet

            Returns:
                binary_sum (str): Binary string of the sum of a and b
        """
        # cls.logger.debug(f"get_result_set_to_list::{cls.__name__=}")
        retData: list[Self] = []

        keys: sqlalchemy.engine.result.RMKeyView = rs.keys()

        while True:
            row: Self|None = cls().from_result_set_next_row(rs, list(keys))
            if row is None:
                break

            retData.append(row)

        return retData

    @classmethod
    def get_list_from_sql(cls, sqlstr: str, **kwargs: Any) -> List[Self]:
        with DBConnectionEngine.get_instance().connect() as conn:
            sql = text(sqlstr)
            rs: sqlalchemy.engine.cursor.Result = conn.execute(sql, **kwargs)
            return cls.get_result_set_to_list(rs)

    @classmethod
    def get_one_from_sql(cls, sqlstr: str, **kwargs: Any) -> Self|None:
        # cls.logger.debug(f"{cls.__name__=}")

        conn: Connection
        with DBConnectionEngine.get_instance().connect() as conn:
            sql = text(sqlstr)

            rs: sqlalchemy.engine.cursor.Result = conn.execute(sql, kwargs)
            keys: sqlalchemy.engine.result.RMKeyView = rs.keys()
            # cls.logger.debug(f"{type(keys)=} {keys=}")

            colnames = list(keys)
            # cls.logger.debug(f"{type(colnames)=} {colnames=}")

            ret = cls().from_result_set_next_row(rs, colnames)
            return ret

    def insertnew(self) -> Optional[DBObjectInsertUpdateDeleteResult]:
        if self.__getattribute__("dbcolnames") is None:
            raise Exception('Object does not have property "dbcolnames"')
        if self.__getattribute__("dbtablename") is None:
            raise Exception('Object does not have property "dbtablename"')

        tablename: str = self.__getattribute__("dbtablename")
        savesql: str = f"INSERT INTO {tablename} ( "
        vsql: str = ""
        vhash: dict = {}

        assert self.dbcolnames
        for index, colname in enumerate(self.dbcolnames):
            if index > 0:
                savesql = savesql + ", "
                vsql = vsql + ", "

            savesql = savesql + colname
            vsql = vsql + f":{colname}"

            v = self.__getattribute__(colname)
            if isinstance(v, datetime.datetime):
                v2: datetime.datetime = v
                v = v2.strftime("%Y-%m-%dT%H:%M:%S.%f")
            elif isinstance(v, StrEnum):
                v3: StrEnum = v
                v = v3.value
            elif isinstance(v, dict) or isinstance(v, list):
                v4: dict | list = v
                v = get_pretty_dict_json_no_sort(v4)
            elif isinstance(v, uuid.UUID):
                v5: uuid.UUID = v
                v = str(v5)

            vhash[colname] = v

        savesql = savesql + ") VALUES (" + vsql + ")"

        if LOGME_VERBOSE:
            self.logger.debug(f"{savesql=} {vhash=}")

        result: DBObjectInsertUpdateDeleteResult|None = self.insertupdatedelete(savesql, **vhash)

        # traceback.print_tb(result.getException())
        # print("Result.exceptionOccured: ", result.exceptionOccured())
        # print("Result.rowsAffected: ", result.getRowsAffected())
        return result
