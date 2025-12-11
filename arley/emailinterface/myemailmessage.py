# https://docs.python.org/3/library/email.examples.html
import datetime
import email
import io
import re
from email import utils
from email.header import decode_header
from email.message import EmailMessage
from typing import Any, List, Self
from uuid import UUID

import html_text
import pytz
from loguru import logger
from lxml.html import HtmlElement
from mailparser_reply import EmailMessage as ReplyEmailMessage
from mailparser_reply import EmailReply, EmailReplyParser

from arley import Helper
from arley.config import settings

_timezone: datetime.tzinfo = pytz.timezone(settings.timezone)

MAIL_LANGUAGES: list[str] = ["en", "de"]
LOGME_VERBOSE: bool = False


class MyEmailMessage:
    logger = logger.bind(classname=__qualname__)

    # headers with relevance:
    #     In-Reply-To
    #     Message-ID:
    #     Return-path
    #     Reply-To

    email_pattern: re.Pattern[str] = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    arley_id_pattern: re.Pattern[str] = re.compile(
        r"\[(arley-id)(\s*)([0-9a-f]{8}-[0-9a-f]{4}-[0-5][0-9a-f]{3}-[089ab][0-9a-f]{3}-[0-9a-f]{12})\]"
    )

    def __init__(
        self,
        msgid: int,
        email_message: EmailMessage,
        envelope_message_id: str,
        from_email: str,
        to_email: str,
        subject: str,
    ):
        self.msgid: int = msgid
        self.email_message: EmailMessage = email_message
        self.parsed_email: ReplyEmailMessage | None = None
        self.envelope_message_id: str = envelope_message_id
        self.from_email: str = from_email
        self.to_email: str = to_email
        self.subject: str = subject
        self.textbody_full: str | None = None

        self._sanitize_header_fields()

    def _sanitize_header_fields(self) -> None:
        if self.envelope_message_id:
            self.envelope_message_id = self.envelope_message_id.replace("\n", "").replace("\r", "")

        if self.from_email:
            self.from_email = self.from_email.replace("\n", "").replace("\r", "")

        if self.to_email:
            self.to_email = self.to_email.replace("\n", "").replace("\r", "")

        if self.subject:
            self.subject = self.subject.replace("\n", "").replace("\r", "")

    @classmethod
    def get_date_from_header(cls, email_message: EmailMessage, dateheader_name: str) -> datetime.datetime | None:
        dhs: list | None = email_message.get_all(dateheader_name)
        if not dhs:
            return None

        try:
            rec: str = dhs[0]
            rec_date_str: str = rec
            if rec.find("; ") > 0:
                rec_date_str = rec[rec.rfind(";") + 1 :].strip()

            rec_date_datetime: datetime.datetime = email.utils.parsedate_to_datetime(rec_date_str)
            rec_date_datetime = rec_date_datetime.astimezone(_timezone)

            return rec_date_datetime
        except Exception as e:
            cls.logger.exception(e)

        return None

    def get_date_from_last_received_header(self) -> datetime.datetime | None:
        return MyEmailMessage.get_date_from_header(self.email_message, "Received")

    def get_date_from_date_header(self) -> datetime.datetime | None:
        return MyEmailMessage.get_date_from_header(self.email_message, "Date")

    def to_json(self) -> dict:
        ret: dict[str, Any] = {}
        ret["msgid"] = self.msgid
        ret["envelope_message_id"] = self.envelope_message_id
        ret["from_email"] = self.from_email
        ret["to_email"] = self.to_email
        ret["subject"] = self.subject
        ret["email_message"] = (
            self.email_message
        )  # bytes(self.email_message.as_string(), "utf-8").decode("unicode_escape")
        # self.logger.debug(ret["email_message"])
        ret["parsed_email"] = self.parsed_email
        ret["original_textbody"] = self.get_textbody()

        return ret

    @classmethod
    def decode_my_header(cls, header_in: Any | None) -> str | None:
        if not header_in:
            return None

        buffer: io.StringIO = io.StringIO()

        # print(f"{LOGME_VERBOSE=}")

        if LOGME_VERBOSE:
            cls.logger.debug(f"decode_my_header::{type(header_in)=} {header_in=}")

        dh: list[tuple[Any, Any | None]] = decode_header(header_in)

        if LOGME_VERBOSE:
            cls.logger.debug(f"decode_my_header::dh:{type(dh)=} {dh=}")

        dh_elem: tuple[Any, Any | None]
        for dh_elem in dh:
            if LOGME_VERBOSE:
                cls.logger.debug(f"decode_my_header::dh_elem:{type(dh_elem)=} {dh_elem=}")

            if isinstance(dh_elem[0], str):
                buffer.write(dh_elem[0])
            else:
                buffer.write(dh_elem[0].decode(dh_elem[1] if dh_elem[1] else "ascii"))

        return buffer.getvalue()

    @classmethod
    def get_textbody_from_email_message_and_update_my_email_message(
        cls, email_message: EmailMessage, my_email_message: Self | None
    ) -> str | None:
        if LOGME_VERBOSE:
            cls.logger.debug(f"{email_message.get_content_type()=}")
            cls.logger.debug(f"{email_message.get_content_maintype()=}")
            cls.logger.debug(f"{email_message.get_content_subtype()=}")

        cs = email_message.get_charset()
        if LOGME_VERBOSE:
            cls.logger.debug(f"{type(cs)=} {cs=}")

        if not cs:
            cs = "UTF-8"

        # msg_body = None
        # # Extract the body of the email
        # if msg.is_multipart():
        #     for part in msg.walk():
        #         # if part.get_content_type() == 'text/html':
        #         if part.get_content_type() == 'text/plain':
        #             msg_body = part.get_payload(decode=True).decode()
        #             break
        # else:
        #     msg_body = msg.get_payload(decode=True).decode()

        textbody: str | None = None

        if LOGME_VERBOSE:
            cls.logger.debug(f"{email_message.is_multipart()=}")
        if email_message.is_multipart():
            for part in email_message.walk():
                if LOGME_VERBOSE:
                    cls.logger.debug(f"{type(part)=}")
                    cls.logger.debug(f"{part.get_content_type()=}")
                    cls.logger.debug(f"{part.get_content_maintype()=}")
                    cls.logger.debug(f"{part.get_content_subtype()=}")

                    # if part.get_content_type().find("image") < 0:
                    #     cls.logger.debug(part)

                # part.get_payload()
                if part.get_content_type().find("text/plain") >= 0:
                    mep: EmailMessage | bytes | Any = part.get_payload(decode=True)
                    csm: str | None = part.get_content_charset()
                    if hasattr(mep, "decode"):
                        textbody = mep.decode(csm if csm else "utf-8").strip()
                elif textbody is None and part.get_content_type().find("text/html") >= 0:
                    # htmltext: str = part.get_payload(decode=True).decode(part.get_content_charset())
                    hmep: EmailMessage | bytes | Any = part.get_payload(decode=True)
                    hcsm: str | None = part.get_content_charset()
                    htmltext: str | None = None
                    if hasattr(hmep, "decode"):
                        htmltext = hmep.decode(hcsm if hcsm else "utf-8").strip()

                    if LOGME_VERBOSE:
                        cls.logger.debug(f"HTMLTEXT:\n{htmltext}\n\n\n")

                    assert htmltext is not None
                    htmltree: HtmlElement = html_text.parse_html(htmltext)
                    cleaned_tree: HtmlElement = html_text.cleaner.clean_html(htmltree)
                    decodedhtml: str = html_text.etree_to_text(cleaned_tree)

                    if LOGME_VERBOSE:
                        cls.logger.debug(f"DECODEDHTML:\n{decodedhtml}")

                    textbody = decodedhtml.strip()
        else:
            # textbody = email_message.get_payload(decode=True).decode(email_message.get_content_charset()).strip()
            tmep: EmailMessage | bytes | Any = email_message.get_payload(decode=True)
            tcsm: str | None = email_message.get_content_charset()
            if hasattr(tmep, "decode"):
                textbody = tmep.decode(tcsm if tcsm else "utf-8").strip()

        if my_email_message:
            my_email_message.textbody_full = textbody
            my_email_message.parsed_email = EmailReplyParser(languages=MAIL_LANGUAGES).read(text=textbody)

        if LOGME_VERBOSE and my_email_message is not None:
            cls.logger.debug(f"TEXTBODY:\n{textbody}\n\n")
            cls.logger.debug(f"PARSED_TEXTBODY:\n{my_email_message.parsed_email}")
            cls.logger.debug(EmailReplyParser(languages=MAIL_LANGUAGES).read(text=textbody).text)

        return textbody

    def ensure_parsed(self) -> None:
        if not self.parsed_email:
            MyEmailMessage.get_textbody_from_email_message_and_update_my_email_message(self.email_message, self)

    def get_in_reply_to(self) -> str | None:
        return MyEmailMessage.decode_my_header(self.email_message.get("In-Reply-To"))

    def get_references(self) -> str | None:
        return MyEmailMessage.decode_my_header(self.email_message.get("References"))

    def get_arley_ids_from_subject_and_text(self) -> UUID | None:
        arleyid_from_subject: UUID | None = MyEmailMessage.get_arley_id(self.subject)
        arleyid_from_text: UUID | None = MyEmailMessage.get_arley_id(self.get_latest_reply())

        if arleyid_from_text and arleyid_from_subject and str(arleyid_from_subject) != str(arleyid_from_text):
            raise RuntimeError(f"arleyid_mismatch::{str(arleyid_from_subject)=} {str(arleyid_from_text)=}")

        return arleyid_from_subject

    @classmethod
    def get_arley_id(cls, search_in_text: str | None) -> UUID | None:
        # ***********************************************************************************
        # please keep this in your reply and do not make changes below the line before this
        # [arley-id {{ ARLEYID }}]
        # ***********************************************************************************
        # two groups enclosed in separate ( and ) bracket

        if search_in_text is None:
            return None

        arley_ids: set[UUID] = set()

        matches: list[str | tuple[str, ...]] | None = cls.arley_id_pattern.findall(search_in_text)
        if matches:
            for match in matches:
                # Extract matching values of all groups
                if LOGME_VERBOSE:
                    cls.logger.debug(f"{type(match)=} {match=}")
                if isinstance(match, tuple):
                    if LOGME_VERBOSE:
                        cls.logger.debug(f"{match[0]=} {match[1]=} {match[2]=}")
                    arley_ids.add(UUID(match[2]))
                else:
                    # str-type
                    if LOGME_VERBOSE:
                        cls.logger.debug(f"{match=}")
                    arley_ids.add(UUID(match))

        if len(arley_ids) == 0:
            return None

        if len(arley_ids) > 1:
            raise RuntimeError(f"Multiple (diverging) arley ids found: {arley_ids=}")

        return arley_ids.pop()

    def get_all_replies(self) -> list[EmailReply]:
        self.ensure_parsed()

        assert self.parsed_email is not None
        replies: list[EmailReply] = self.parsed_email.replies

        # for i in replies:
        #     self.logger.debug(f"{i.content=}")
        #     self.logger.debug(f"{i.signatures=}")
        #     self.logger.debug(f"{i.disclaimers=}")

        return replies

    def print_info(self) -> None:
        pinfo: dict = {}
        for k, v in self.email_message.items():
            # self.logger.debug(f"{k=}\t{v=}")
            pinfo[k] = v

        logger.debug(Helper.get_pretty_dict_json_no_sort(pinfo))

    def get_textbody(self) -> str | None:
        self.ensure_parsed()

        return self.textbody_full

    def get_latest_reply(self) -> str | None:
        self.ensure_parsed()

        assert self.parsed_email is not None

        if not self.parsed_email.replies:
            return None

        latest_reply: str = self.parsed_email.replies[0].body

        if LOGME_VERBOSE:
            dm: dict = {
                "reply_body": latest_reply,
                "reply_full_body": self.parsed_email.replies[0].full_body,
                "reply_headers": self.parsed_email.replies[0].headers,
                "reply_disclaimers": self.parsed_email.replies[0].disclaimers,
            }
            self.logger.debug(Helper.get_pretty_dict_json_no_sort(dm))

        return latest_reply

    # msg = BytesParser(policy=policy.default).parse(fp)
    #
    # # Now the header items can be accessed as a dictionary, and any non-ASCII will
    # # be converted to unicode:
    # print('To:', msg['to'])
    # print('From:', msg['from'])
    # print('Subject:', msg['subject'])
    #
    # # If we want to print a preview of the message content, we can extract whatever
    # # the least formatted payload is and print the first three lines.  Of course,
    # # if the message has no plain text part printing the first three lines of html
    # # is probably useless, but this is just a conceptual example.
    # simplest = msg.get_body(preferencelist=('plain', 'html'))
    # print()
    # print(''.join(simplest.get_content().splitlines(keepends=True)[:3]))
    #
    # ans = input("View full message?")
    # if ans.lower()[0] == 'n':
    #     sys.exit()
    #
    # # We can extract the richest alternative in order to display it:
    # richest = msg.get_body()
    # partfiles = {}
    # if richest['content-type'].maintype == 'text':
    #     if richest['content-type'].subtype == 'plain':
    #         for line in richest.get_content().splitlines():
    #             print(line)
    #         sys.exit()
    #     elif richest['content-type'].subtype == 'html':
    #         body = richest
    #     else:
    #         print("Don't know how to display {}".format(richest.get_content_type()))
    #         sys.exit()
    # elif richest['content-type'].content_type == 'multipart/related':
    #     body = richest.get_body(preferencelist=('html'))
    #     for part in richest.iter_attachments():
    #         fn = part.get_filename()
    #         if fn:
    #             extension = os.path.splitext(part.get_filename())[1]
    #         else:
    #             extension = mimetypes.guess_extension(part.get_content_type())
    #         with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as f:
    #             f.write(part.get_content())
    #             # again strip the <> to go from email form of cid to html form.
    #             partfiles[part['content-id'][1:-1]] = f.name
    # else:
    #     print("Don't know how to display {}".format(richest.get_content_type()))
    #     sys.exit()
    # with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    #     f.write(magic_html_parser(body.get_content(), partfiles))
    # webbrowser.open(f.name)
    # os.remove(f.name)
    # for fn in partfiles.values():
    #     os.remove(fn)


def do_myemail_test() -> None:
    from arley.emailinterface.imapadapter import IMAPAdapter

    ima: IMAPAdapter = IMAPAdapter()
    ima.login()

    try:
        from imapclient.response_types import Envelope

        mails: list[tuple[int, Envelope]] = ima.list_mails(settings.emailsettings.folders.cur)
        for msgid, env in mails:
            logger.debug(f"{msgid=} {env=}")
            env_messageid: str = env.message_id.decode().strip()

            my_email_message: MyEmailMessage | None = ima.get_message(
                msgid=msgid, folder=settings.emailsettings.folders.cur
            )
            if my_email_message is None:
                continue

            email_message: EmailMessage = my_email_message.email_message
            # EmailReplyParser(languages=MAIL_LANGUAGES).read(text=textbody)
            # my_email_message.ensure_parsed()
            textbody: str | None = my_email_message.get_textbody_from_email_message_and_update_my_email_message(
                email_message=email_message, my_email_message=my_email_message
            )
            logger.debug(textbody)

    except Exception as ex:
        logger.exception(ex)
    finally:
        ima.logout()

    #         if LOGME_VERBOSE:
    #             cls.logger.debug(f"{email_message.get_content_type()=}")
    #             cls.logger.debug(f"{email_message.get_content_maintype()=}")
    #             cls.logger.debug(f"{email_message.get_content_subtype()=}")
    #
    #         cs = email_message.get_charset()
    #         if LOGME_VERBOSE:
    #             cls.logger.debug(f"{type(cs)=} {cs=}")
    #
    #         if not cs:
    #             cs = "UTF-8"
    #
    #         # msg_body = None
    #         # # Extract the body of the email
    #         # if msg.is_multipart():
    #         #     for part in msg.walk():
    #         #         # if part.get_content_type() == 'text/html':
    #         #         if part.get_content_type() == 'text/plain':
    #         #             msg_body = part.get_payload(decode=True).decode()
    #         #             break
    #         # else:
    #         #     msg_body = msg.get_payload(decode=True).decode()
    #
    #         textbody: str | None = None
    #
    #         if LOGME_VERBOSE:
    #             cls.logger.debug(f"{email_message.is_multipart()=}")
    #         if email_message.is_multipart():
    #             for part in email_message.walk():
    #                 if LOGME_VERBOSE:
    #                     cls.logger.debug(f"{type(part)=}")
    #                     cls.logger.debug(f"{part.get_content_type()=}")
    #                     cls.logger.debug(f"{part.get_content_maintype()=}")
    #                     cls.logger.debug(f"{part.get_content_subtype()=}")
    #
    #                     if part.get_content_type().find("image") < 0:
    #                         cls.logger.debug(part)
    #
    #                 # part.get_payload()
    #                 if part.get_content_type().find("text/plain") >= 0:
    #                     textbody = part.get_payload(decode=True).decode(part.get_content_charset())
    #         else:
    #             textbody = email_message.get_payload(decode=True).decode(email_message.get_content_charset())
    #
    #         if my_email_message:
    #             my_email_message.textbody_full = textbody
    #             my_email_message.parsed_email = EmailReplyParser(languages=MAIL_LANGUAGES).read(text=textbody)
    #
    #         if LOGME_VERBOSE:
    #             cls.logger.debug(f"{textbody=}")
    #
    #         return textbody

    #


if __name__ == "__main__":
    do_myemail_test()
