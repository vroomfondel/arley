import datetime
import email
import json
from email import utils
from email.message import EmailMessage
from pprint import pprint
from typing import Any

import pytz
from imapclient.response_types import Envelope
from loguru import logger
from ollama import Message

from arley import Helper
from arley.config import is_in_cluster, settings
from arley.dbobjects.emailindb import ArleyEmailInDB, ArleyRawEmailInDB
from arley.emailinterface.imapadapter import IMAPAdapter
from arley.emailinterface.myemailmessage import MyEmailMessage
from arley.emailinterface.ollamaemailreply import OllamaEmailReply

_timezone: datetime.tzinfo = pytz.timezone(settings.timezone)


def adapt_emails_ollama_msgs() -> None:
    # generate_msgs_for_ollama
    # select length(ollamamsgs::text),emailid,ollamamsgs from emails where length(ollamamsgs::text)>2;
    emailsindb_from_arley: list[ArleyEmailInDB] = ArleyEmailInDB.get_list_from_sql(
        "select * from emails where fromarley and processresult='processed' order by received desc"
    )
    #  where select distinct rootemailid, received from emails where fromarley order by received desc, rootemailid
    for emailindb in emailsindb_from_arley:
        previous_texts: list[tuple[str, bool]] = []

        prev_sql: str = (
            f"select * from emails where rootemailid='{emailindb.rootemailid}' and sequencenumber<{emailindb.sequencenumber} order by sequencenumber asc"
        )
        logger.debug(f"{prev_sql=}")

        previous: list[ArleyEmailInDB] = ArleyEmailInDB.get_list_from_sql(prev_sql)

        initial_topic = previous[0].subject
        initial_prompt = previous[0].mailbody

        for prev in previous:
            assert prev.mailbody is not None
            assert prev.fromarley is not None
            previous_texts.append((prev.mailbody, prev.fromarley))

        # das hier ist ja strenggenommen die antwort -> in mailindb.mailbody drin
        # previous_texts.append((emailindb.mailbody, emailindb.fromarley))

        assert initial_topic
        msgs: list[Message] = OllamaEmailReply.generate_msgs_for_ollama(
            lang="de", previous_texts=previous_texts, initial_topic=initial_topic
        )
        logger.debug(msgs)
        # email_message: EmailMessage = email.message_from_string(rawemail.rawemail)  # type: ignore

        emailindb.ollamamsgs = msgs
        emailindb.save()
        # if msgs:
        #     break


def adapt_rawemails() -> None:
    rawemailsindb: list[ArleyRawEmailInDB] = ArleyRawEmailInDB.get_list_from_sql("select * from rawemails")
    for rawemail in rawemailsindb:
        email_message: EmailMessage = email.message_from_string(rawemail.rawemail)  # type: ignore
        msgid_raw = email_message.get("Message-ID")
        # msgid = utils.collapse_rfc2231_value(msgid_raw)
        logger.debug(f"{'*'*50}")
        # logger.debug(f"{msgid_raw=}\t{msgid=}")
        logger.debug(f"{msgid_raw=}")

        # mailbody_text: str | None = MyEmailMessage.get_textbody_from_email_message_and_update_my_email_message(
        #     email_message=email_message,
        #     my_email_message=None
        # )
        #
        # rawemail.textmailbody = mailbody_text
        # rawemail.save()

        # rawparam = msg. get_param('foo')
        # param = email. utils. collapse_rfc2231_value(rawparam)

        rd: datetime.datetime | None = MyEmailMessage.get_date_from_header(
            email_message=email_message, dateheader_name="Received"
        )
        logger.debug(f"Received: {type(rd)=}\t{rd=}")

        rdd: datetime.datetime | None = MyEmailMessage.get_date_from_header(
            email_message=email_message, dateheader_name="Date"
        )
        logger.debug(f"Date: {type(rdd)=}\t{rdd=}")

        aeid: ArleyEmailInDB | None = ArleyEmailInDB.get_one_from_sql(
            f"select * from emails where emailid='{rawemail.emailid}'"
        )

        assert aeid
        aeid.received = rd
        if not aeid.received:
            aeid.received = rdd

        aeid.save()


def check() -> None:
    ima: IMAPAdapter = IMAPAdapter()
    ima.login()

    try:
        folders: list[str] = ima.list_folders(folder="WORKED")
        for folder in folders:
            logger.debug(f"{folder=}")
            mails: list[tuple[int, Envelope]] = ima.list_mails(folder=folder)
            for msgid, env in mails:
                logger.debug(f"{'*'*50}")
                logger.debug(f"{'*' * 50}")
                logger.debug(f"{msgid=}")
                logger.debug(env)
                myemail: MyEmailMessage | None = ima.get_message(msgid, folder=folder)
                if not myemail:
                    continue

                myemail.get_in_reply_to()
                # logger.debug(Helper.get_pretty_dict_json_no_sort())

                myemail.ensure_parsed()
                jso: dict = myemail.to_json()

                # em: str = jso["email_message"]
                # del jso["email_message"]
                # bytes(myString, "utf-8").decode("unicode_escape")
                # jso["email_message"] = self.email_messagejso["email_message"]

                logger.debug(f"{myemail.get_references()=}")
                logger.debug(Helper.get_pretty_dict_json_no_sort(jso))

                # myemail._ensure_parsed()
                #
                # for reply in myemail.parsed_email.replies:
                #     logger.debug(f"{reply=}")
                #     logger.debug(reply.body)
                # logger.debug(em)

                logger.debug(f"{myemail.get_latest_reply()=}")

                logger.debug(myemail.get_textbody())

    except Exception as e:
        logger.exception(e)
    finally:
        ima.logout()

    return None


if __name__ == "__main__":
    adapt_emails_ollama_msgs()
    # adapt_rawemails()
    # check()
