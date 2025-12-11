from typing import Optional
from uuid import uuid4, UUID

import sqlalchemy.inspection
from sqlalchemy.dialects.postgresql.base import PGInspector

# import dbconnection
from sqlalchemy import inspect, select

from sqlalchemy import MetaData
from sqlalchemy.sql.schema import Table

# https://docs.sqlalchemy.org/en/14/core/reflection.html#reflecting-all-tables-at-once


def as_valid_uuid4(uuid_to_test: str) -> Optional[UUID]:
    try:
        ret: UUID = UUID(uuid_to_test, version=4)
        return ret
    except ValueError:
        return None


def main() -> None:
    ...


# meta = MetaData()
# insp: PGInspector = inspect(dbconnection.psqlengine)
# for tablename in insp.get_table_names():
#     print(f"tablename({type(tablename)}: {tablename}") # str
#
#     t: Table = Table(tablename, meta, keep_existing=True, autoload_with=dbconnection.psqlengine)
#     s: select = select([t])._all()
#
#     # s = select([users.c.name, users.c.fullname])
#     # SQL >> > result = conn.execute(s)
#     #t.select(t.c.)
#     # >>> s = select([users, addresses]).where(users.c.id == addresses.c.user_id)
#     # print(users.c.id == addresses.c.user_id)
#
#     # >>> s = select([(users.c.fullname +
#     # ...               ", " + addresses.c.email_address).
#     # ...                label('title')]).\
#     # ...        where(
#     # ...           and_(
#     # ...               users.c.id == addresses.c.user_id,
#     # ...               users.c.name.between('m', 'z'),
#     # ...               or_(
#     # ...                  addresses.c.email_address.like('%@aol.com'),
#     # ...                  addresses.c.email_address.like('%@msn.com')
#     # ...               )
#     # ...           )
#     # ...        )
#     # >>> conn.execute(s).fetchall()
#
#     print(f"t: {type(t)}")
#     help(t)
#
#     for columname in insp.get_columns(tablename):
#         print(f"\tcolumname({type(columname)}: {columname}")  # dict
#     break

# print(f"{type(insp)}")  # sqlalchemy.dialects.postgresql.base.PGInspector

# meta = MetaData()
# meta.reflect(bind=dbconnection.psqlengine)
# for table in meta.sorted_tables:
#     table: Table = table
#     #table.name
#     table.get
#     print(f"table(name={table.name}, type={type(table)}")
#     help(table)
#     break


if __name__ == "__main__":
    main()
