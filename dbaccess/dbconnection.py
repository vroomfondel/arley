import sys
import datetime
import traceback
import uuid
from contextlib import contextmanager
from pprint import pprint
from typing import List, Optional, Any, Self, Callable, Generator, Dict

import loguru
import sqlalchemy
from sqlalchemy import create_engine, event, text, exc, Engine, URL, Connection
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.sql import text
import os

# https://chartio.com/resources/tutorials/how-to-execute-raw-sql-in-sqlalchemy/
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.dialects import postgresql
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey
from sqlalchemy import inspect

from loguru import logger

import arley.config
from arley.Helper import Singleton

LOGME_VERBOSE: bool = False


class DBConnectionEngine(metaclass=Singleton):
    logger = logger.bind(classname=__qualname__)

    def __init__(self) -> None:
        super().__init__()

        self.psqlengine: Engine | None = None
        self._init_connection(self._get_connection_url_from_env())

    def _get_connection_url_from_env(self) -> URL:
        self.logger.debug("_get_connection_url_from_env() called")
        dburl: URL

        if "PSQL_DB_URL" in os.environ:
            dburl = sqlalchemy.engine.url.make_url(os.environ["PSQL_DB_URL"])
        else:
            db_user: str = os.environ["PSQL_DB_USERNAME"]
            db_pass: str = os.environ["PSQL_DB_PASSWORD"]
            db_name: str = os.environ["PSQL_DB_NAME"]
            db_host: str = os.environ["PSQL_DB_HOST"]
            db_port: int = int(os.environ["PSQL_DB_PORT"])

            dburl = sqlalchemy.engine.url.URL.create(
                drivername="postgresql+psycopg2",
                username=db_user,
                password=db_pass,
                database=db_name,
                host=db_host,
                port=db_port,
                # query={"host": unix_socket_path},  # if going via socket.
            )

        return dburl

    def _init_connection(self, dburl: URL) -> None:
        self.logger.debug(f"{dburl=}")

        self.psqlengine = create_engine(
            dburl,
            poolclass=QueuePool,
            pool_recycle=10,
            # very short_lived pool-maxtime: 10s -> pooling is done on proxysql-level THEN
            pool_size=0,  # means UNLIMITED
            # pool_size=20,
            # max_overflow=0
            echo=True,
            # isolation_level="AUTOCOMMIT",
            query_cache_size=0
        )

    @contextmanager
    def connect(self) -> Generator[Connection, Any, None]:
        connection: Connection = self.get_engine(autocommit=True).connect()
        # connection = connection.execution_options()
        try:
            yield connection
            # connection.commit()
        finally:
            connection.close()

    @classmethod
    def get_instance(cls) -> "DBConnectionEngine":
        return DBConnectionEngine()  # is singleton

    def get_engine(self, autocommit: bool = True) -> Engine:
        isolation_level: str = "AUTOCOMMIT"
        if not autocommit:
            isolation_level = "SERIALIZABLE"

        assert self.psqlengine
        return self.psqlengine.execution_options(isolation_level=isolation_level)


# # THIS checks for connections only accessed within the same process-id (should not be necessary since we are spawning
# # the gunicorn-processes off before the db-pool is setup -> in each process, a new db-pool will be instantiated, but
# # when using multiple processes in cron-stuff could make this necessary/more safe
#
# @event.listens_for(psqlengine, "connect")
# def connect(dbapi_connection, connection_record):
#     connection_record.info['pid'] = os.getpid()
#
# @event.listens_for(psqlengine, "checkout")
# def checkout(dbapi_connection, connection_record, connection_proxy):
#     pid = os.getpid()
#     if connection_record.info['pid'] != pid:
#         connection_record.connection = connection_proxy.connection = None
#         raise exc.DisconnectionError(
#                 "Connection record belongs to pid %s, "
#                 "attempting to check out in pid %s" %
#                 (connection_record.info['pid'], pid)
#         )


def get_exception_tb_as_string(exc: Exception) -> str:
    tb1: traceback.TracebackException = traceback.TracebackException.from_exception(exc)
    tbsG = tb1.format()
    tbs = ""

    for line in tbsG:
        tbs = tbs + "\n" + line

    return tbs

import loguru
def eprint(
        lm: loguru._logger.Logger,  # type: ignore
        *args: Any,
        **kwargs: Any) -> None:
    if not lm:
        print(*args, file=sys.stderr, **kwargs)
    else:
        lm.error(*args, **kwargs)


class DBObjectInsertUpdateDeleteResult:
    logger = logger.bind(classname=__qualname__)

    def __init__(self) -> None:
        self.rowcount: Optional[int] = None
        self.exception: Exception|None = None
        super().__init__()

    def from_result_set(self, rs: sqlalchemy.engine.cursor.CursorResult, ex: Exception|None = None) -> Optional[Self]:
        if ex is not None:
            eprint(DBObjectInsertUpdateDeleteResult.logger, "Exception caught: ", ex)
            self.exception = ex
            raise ex

        if rs is not None:
            self.rowcount = rs.rowcount

        return self

    def exception_occured(self) -> bool:
        return self.exception is not None

    def get_exception(self) -> Exception|None:
        return self.exception

    def get_rows_affected(self) -> Optional[int]:
        return self.rowcount


class MultiLineResSet:
    def __init__(self, colnames: List[str], data_ref: Optional[List[List[Any]]] = None):
        self.colnames = colnames
        self.data: List[Any]
        if data_ref is not None:
            self.data = data_ref
        else:
            self.data = []

        super(MultiLineResSet, self).__init__()

    def get_row_count(self) -> int:
        return len(self.data)

    def add_line(self, row: List[Any]) -> None:
        assert len(self.colnames) == len(row)

        self.data.append(row)

    def get_row(self, rownum: int) -> List[Any]:
        assert rownum >= 0 and rownum < len(self.data)
        return self.data[rownum]

    def get_value_at(self, rownum: int, colnum: int) -> Any:
        row = self.get_row(rownum)
        assert colnum >= 0 and colnum < len(row)
        return row[colnum]

    def get_value_of(self, rownum: int, colname: str) -> Any:
        row = self.get_row(rownum)
        colnum = self.colnames.index(colname)
        assert colnum is not None and colnum >= 0
        return row[colnum]

    def get_values_of(self, colname: str) -> List:
        colnum = self.colnames.index(colname)
        assert colnum is not None and colnum >= 0

        ret: list = []
        for j in range(0, self.get_row_count()):
            row = self.get_row(j)
            ret.append(row[colnum])
        return ret

    def get_all_rows_as_list_of_dicts(self) -> List[dict]:
        ret: list = []
        for j in range(0, self.get_row_count()):
            retrow: dict = {}
            row = self.get_row(j)
            for i in range(0, len(self.colnames)):
                retrow[self.colnames[i]] = row[i]
            ret.append(retrow)
        return ret

    def get_row_as_dict(self, rownum: int) -> dict:
        ret: dict = {}
        row = self.get_row(rownum)
        for i in range(0, len(self.colnames)):
            ret[self.colnames[i]] = row[i]
        return ret

    def get_lookup_dict(self, keycolumnname: str) -> Dict:
        """
        kinda assumes, that keycolumname is UNIQUE
        """

        _ret: dict = {}

        for j in range(0, self.get_row_count()):
            fn: str = self.get_value_of(j, keycolumnname)
            _ret[fn] = j

        return _ret

    def get_row_lookup_dict(self, keycolumnname: str) -> Dict:
        """
        kinda assumes, that keycolumname is UNIQUE
        """

        _ret: dict = {}

        for j in range(0, self.get_row_count()):
            fn: str = self.get_value_of(j, keycolumnname)
            _ret[fn] = self.get_row_as_dict(j)

        return _ret

    def get_rows_lookup_dict(self, keycolumnname: str) -> Dict:
        _ret: dict = {}

        for j in range(0, self.get_row_count()):
            fn: str = self.get_value_of(j, keycolumnname)
            _ret[fn] = _ret.get(fn) or []
            _ret[fn].append(self.get_row_as_dict(j))

        return _ret

    def find_rows(self, params: Dict) -> List:
        ret: list = []
        for j in range(0, self.get_row_count()):
            for columnname in params:
                if self.get_value_of(j, columnname) != params[columnname]:
                    continue

            retrow: dict = {}
            row = self.get_row(j)
            for i in range(0, len(self.colnames)):
                retrow[self.colnames[i]] = row[i]

            ret.append(retrow)
        return ret



def get_compiled_sql(sql: str, **bindparams: Any) -> str:
    """Compile a SQL text with ":variable" placeholders into a literal SQL string
    using the PostgreSQL dialect.

    Example:
        get_compiled_sql("select :x::int as v", x=5) -> "SELECT 5::INTEGER AS v"
    """

    # check also:
    # https://stackoverflow.com/questions/5631078/sqlalchemy-print-the-actual-query
    # https://stackoverflow.com/questions/5631078/sqlalchemy-print-the-actual-query/5698357#5698357

    # return LiteralCompiler(dialect=dialect).process(sql_stmt)

    # sql_stmt = sql_stmt.bindparams(**bindparams)
    # sql_filled = text(sql).bindparams(**bindparams)

    # return sql_stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    # print("Compiled-SQL-with-args: ", sql_filled.compile(dialect=conn.dialect, compile_kwargs={"literal_binds": True}))


    # Build a TextClause with bound parameters
    sql_stmt: TextClause = text(sql).bindparams(**bindparams)

    # Compile with the PostgreSQL dialect and render literal values in-place
    compiled = sql_stmt.compile(
        dialect=postgresql.dialect(),
        compile_kwargs={"literal_binds": True},
    )

    return str(compiled)





def exec_one_line(sqlstr: str, **kwargs: Any) -> Optional[dict]:
    sql: TextClause = text(sqlstr)
    conn: Connection
    with DBConnectionEngine.get_instance().connect() as conn:
        rs: sqlalchemy.engine.cursor.Result = conn.execute(sql, **kwargs)
        keys: sqlalchemy.engine.result.RMKeyView = rs.keys()
        colnames: list = list(keys)
        ret: dict|None = None

        while True:
            row = rs.fetchone()
            logger.debug(f"colnames(type={type(colnames)=}: {colnames=}")
            # print(f"row(type={type(row)}: {row}")
            if row is None:
                break

            if ret is None:
                ret = {}

            # print(f"ROW: {row}")
            for ind, c in enumerate(row):
                ret[colnames[ind]] = c
                # LoggingHelper.debug(f"\t[{ind} type={type(c)} name={colnames[ind]}]: {c}")

    return ret


def exec_multi_line(sqlstr: str, **kwargs: Any) -> Optional[MultiLineResSet]:
    sql = text(sqlstr)
    with DBConnectionEngine.get_instance().connect() as conn:
        rs: sqlalchemy.engine.cursor.Result = conn.execute(sql, **kwargs)
        keys: sqlalchemy.engine.result.RMKeyView = rs.keys()
        colnames: list = list(keys)
        ret: MultiLineResSet|None = None

        while True:
            if ret is None:
                ret = MultiLineResSet(colnames)

            row = rs.fetchone()
            if row is None:
                break

            # print(f"ROW: {row}")
            addrow: List[Any] = []
            for ind, c in enumerate(row):
                addrow.append(c)
                # ret[colnames[ind]] = c
                # LoggingHelper.debug(f"\t[{ind} type={type(c)} name={colnames[ind]}]: {c}")

            ret.add_line(addrow)

    return ret


def insertupdatedelete(sqlstr: str, **kwargs: Any) -> Optional[DBObjectInsertUpdateDeleteResult]:
    # https://chartio.com/resources/tutorials/how-to-execute-raw-sql-in-sqlalchemy/
    # https://stackoverflow.com/questions/36524293/inserting-into-postgres-database-from-sqlalchemy

    conn: sqlalchemy.engine.base.Connection
    rs: sqlalchemy.engine.cursor.CursorResult | None = None
    ex: Exception|None = None
    with DBConnectionEngine.get_instance().connect() as conn:
        try:
            sql = text(sqlstr)
            rs = conn.execute(sql, **kwargs)
            # print("RowCount: ", rs.rowcount)
            # print(f"Type of RS: {type(rs)}")
            # print(f"RS: {rs}")
        except Exception as exX:
            ex = exX

        assert rs
        ret: DBObjectInsertUpdateDeleteResult|None = DBObjectInsertUpdateDeleteResult().from_result_set(rs, ex)

        return ret


def connect_test() -> None:
    logger.info(exec_one_line("select 1=1"))


if __name__ == "__main__":
    print(get_compiled_sql("select :x as v", x=5))
    connect_test()

