import difflib
import sys
import textwrap

import json
import time
import uuid
from email import utils
from email.headerregistry import Address
from typing import Literal, Optional, Tuple, TextIO, Mapping, Any, Iterator

from time import perf_counter


import chromadb
from imapclient.response_types import Envelope

from arley import Helper
from arley.config import (settings,
                          OllamaPrimingMessage,
                          TemplateType,
                          OLLAMA_MODEL,
                          OLLAMA_FUNCTION_CALLING_MODEL,
                          OLLAMA_GUESS_LANGUAGE_MODEL,
                          CHROMADB_DEFAULT_COLLECTION_NAME,
                          ARLEY_AUG_UNIFIED,
                          ARLEY_AUG_PER_ITEM,
                          ARLEY_AUG_NUM_DOCS,
                          ARLEY_AUG_TEMPLATE_TYPE,
                          ARLEY_AUG_ONLY_CONTRACTS,
                          ARLEY_AUG_LANG_FILTER,
                          ARLEY_AUG_FIRST_REQUEST_INCLUDE_AUG,
                          ARLEY_AUG_FIRST_REQUEST_UNIFIED,
                          ARLEY_AUG_FIRST_REQUEST_PER_ITEM,
                          ARLEY_AUG_FIRST_REQUEST_TEMPLATE_TYPE,
                          ARLEY_AUG_FIRST_REQUEST_N_AUG_RESULTS,
                          ARLEY_AUG_FIRST_REQUEST_AUG_ONLY_CONTRACTS,
                          ARLEY_AUG_FIRST_REQUEST_AUG_LANG_FILTER,
                          REFINELOG_RECIPIENTS, OLLAMA_HOST, get_ollama_options)

import datetime
import os
from io import StringIO
from pathlib import Path

import pytz
from jinja2 import Template, Environment, BaseLoader, FileSystemLoader

from reputils import MailReport

from arley.dbobjects.emailindb import ArleyEmailInDB, ArleyRawEmailInDB, Result
from arley.emailinterface.imapadapter import IMAPAdapter
from arley.emailinterface.myemailmessage import MyEmailMessage
from arley.llm.language_guesser import LanguageGuesser

from arley.llm.ollama_adapter import Message, ask_ollama_chat


from arley.llm.ollama_chromadb_rag import OllamaChromaDBRAG
from arley.vectorstore.chroma_adapter import ChromaDBConnection
from dbaccess.db_object import DBObject
from dbaccess.dbconnection import DBObjectInsertUpdateDeleteResult

from chromadb.api.models.Collection import Collection as ChromaCollection

_timezone: datetime.tzinfo = pytz.timezone(settings.timezone)

from loguru import logger

_templatedirpath: Path = Path(__file__).parent.resolve()
_templatedirpath = Path(_templatedirpath, "mailtemplates")

nosend: bool = os.getenv("NOSEND", "False") == "True"

# os.environ["MAILCCS"] = ""

_sdfD_formatstring: str = "%d.%m.%Y"
_sdfDHM_formatstring: str = "%d.%m.%Y %H:%M"
_sdfE_formatstring: str = "%Y%m%d"


class OllamaEmailReply:
    logger = logger.bind(classname=__qualname__)

    def __init__(self, mailindb: ArleyEmailInDB):
        self.mailindb = mailindb

    @classmethod
    def generate_response_from_ollama(cls,
                          is_initial_request: bool,
                          msgs: list[Message],
                          initial_topic: str,
                          prompt: str,
                          # existing_response: str,
                          lang: Literal["en", "de"],
                          primer: list[Message],
                          ollama_model: str = OLLAMA_MODEL,
                          streamed: bool=False,
                          temperature: float=0.1,
                          n_aug_results: int = ARLEY_AUG_NUM_DOCS,
                          template_type: TemplateType = ARLEY_AUG_TEMPLATE_TYPE,
                          unified_aug: bool = ARLEY_AUG_UNIFIED,
                          refine_per_item: bool = ARLEY_AUG_PER_ITEM,
                          aug_only_contracts: bool = ARLEY_AUG_ONLY_CONTRACTS,
                          aug_lang_filter: bool = ARLEY_AUG_LANG_FILTER,
                          first_request_unified: bool=ARLEY_AUG_FIRST_REQUEST_UNIFIED,
                          first_request_per_item: bool=ARLEY_AUG_FIRST_REQUEST_PER_ITEM,
                          first_request_aug_lang_filter: bool=ARLEY_AUG_FIRST_REQUEST_AUG_LANG_FILTER,
                          first_request_include_aug: bool=ARLEY_AUG_FIRST_REQUEST_INCLUDE_AUG,
                          first_request_template_type: TemplateType=ARLEY_AUG_FIRST_REQUEST_TEMPLATE_TYPE,
                          first_request_n_aug_results: int=ARLEY_AUG_FIRST_REQUEST_N_AUG_RESULTS,
                          first_request_aug_only_contracts: bool=ARLEY_AUG_FIRST_REQUEST_AUG_ONLY_CONTRACTS,

                          print_msgs: bool = True,
                          print_responses: bool = True,
                          refinelog: StringIO|TextIO|None = None) -> Tuple[str, dict]:

        logger = cls.logger

        first_request_refined_prompt: str | None = None

        cdbconnection: ChromaDBConnection = ChromaDBConnection.get_instance()

        cdbcollection: ChromaCollection = cdbconnection.get_or_create_collection(CHROMADB_DEFAULT_COLLECTION_NAME)

        rag: OllamaChromaDBRAG = OllamaChromaDBRAG(cdbcollection=cdbcollection)
        
        if is_initial_request and first_request_include_aug:
            # first request
            first_prompt_refines: list[Tuple[str, str]] = rag.get_refine_contexts(
                prompt=prompt,
                lang=lang, # type: ignore
                n_aug_results=first_request_n_aug_results,
                template_type=first_request_template_type,
                refine_per_item=first_request_per_item,
                unified_aug=first_request_unified,
                aug_only_contracts=first_request_aug_only_contracts,
                aug_lang_filter=first_request_aug_lang_filter,
                is_initial_aug_request=True,
                refinelog=refinelog
            )

            logger.debug(f"{len(first_prompt_refines)=}")

            assert len(first_prompt_refines) <= 1, f"undefined behaviour {len(first_prompt_refines)=}>1"
            # first request mit NICHT unified_aug -> macht keinen sinn...
            # => undefined behaviour if there is MORE than one ELEMENT!!!

            for refine_type, merefine in first_prompt_refines:
                first_request_refined_prompt = rag.get_refine_prompt(
                    original_prompt=prompt,
                    existing_response=None,
                    context=merefine,
                    lang=lang,  # type: ignore
                    template_type=first_request_template_type,
                    ollama_model=ollama_model,
                    is_initial_aug_request=True
                )

            logger.debug(f"FIRST REQUEST AUG INITIAL PROMPT:\n{textwrap.indent(prompt, "  PF  ")}")
            if first_request_refined_prompt is not None:
                logger.debug(f"FIRST REQUEST AUG REFINED PROMPT:\n{textwrap.indent(first_request_refined_prompt, "  PR  ")}")

            if refinelog:
                refinelog.write(f"\nFIRST REQUEST AUG INITIAL PROMPT:\n{textwrap.indent(prompt, "  PF  ")}\n\n")
                if first_request_refined_prompt is not None:
                    refinelog.write(f"\nFIRST REQUEST AUG REFINED PROMPT:\n{textwrap.indent(first_request_refined_prompt, "  PR  ")}\n\n")


        resp:  dict | Mapping[str, Any] | Iterator[Mapping[str, Any]] = ask_ollama_chat(
            streamed=streamed,
            system_prompt=None,  # need to provide msg_history then
            prompt=first_request_refined_prompt if first_request_refined_prompt else prompt,
            msg_history=msgs,
            model=ollama_model,
            evict=False,
            temperature=temperature,
            # penalize_newline=True,
            # repeat_penalty=1.0,
            top_k=40,
            top_p=0.9,
            num_predict=-1,
            seed=0,
            print_msgs=print_msgs,
            print_response=print_responses,
        )

        assert isinstance(resp, dict)
        _new_text: str = resp["message"]["content"]

        new_text_wo_think_tag: Optional[str]
        new_text_think_tag: Optional[str]
        new_text_wo_think_tag, new_text_think_tag = Helper.detach_think_tag(_new_text)

        assert new_text_wo_think_tag
        noteline: Optional[str]
        new_text_wo_think_tag, noteline = Helper.detach_NOTE_line(new_text_wo_think_tag)

        if noteline:
            logger.debug(f"INITIAL NOTELINE: {noteline}")
            if refinelog:
                refinelog.write(f"INITIAL NOTELINE: {noteline}\n")

        msg_history_for_refine: list[Message] | None = None

        is_history_mode: bool = False
        if ARLEY_AUG_TEMPLATE_TYPE.value.endswith("chat"):
            is_history_mode = True
            msg_history_for_refine = []
            msg_history_for_refine.extend(msgs)  # da ist die priming-msg mit drin!

            msg_history_for_refine.append(Message(role="user", content=prompt))
            msg_history_for_refine.append(Message(role="assistant", content=new_text_wo_think_tag))

        # this loops internally over all "matched" refines (limited by n_aug and such)...
        new_text_refined: str = rag.refine_response(
            streamed=streamed,
            initial_topic=initial_topic,
            prompt=prompt,
            lang=lang if lang else "en",
            temperature=temperature,
            existing_response=new_text_wo_think_tag,
            primer=msg_history_for_refine if msg_history_for_refine else primer,
            ollama_model=OLLAMA_MODEL,
            refinelog=refinelog,
            unified_aug=unified_aug,
            refine_per_item=refine_per_item,
            n_aug_results=n_aug_results,
            template_type=template_type,
            aug_only_contracts=aug_only_contracts,
            aug_lang_filter=aug_lang_filter,
            is_initial_aug_request=False,
            is_history_mode=is_history_mode
        )

        return new_text_refined, resp

    @classmethod
    def generate_msgs_for_ollama(cls, lang: Literal["de", "en"] | None, previous_texts: list[tuple[str, bool]], initial_topic: str) -> list[Message]:
        msgs: list[Message] = []

        priming_msg: OllamaPrimingMessage
        for priming_msg in settings.ollama.ollama_priming_messages:
            if priming_msg.lang != lang:
                continue
            msgs.append(
                Message(
                    role=priming_msg.role,
                    content=priming_msg.content.replace("$INITIAL_TOPIC", initial_topic)
                )
            )


        for prev_mailbody, prev_fromarley in previous_texts:
            if prev_fromarley:
                msgs.append(Message(role="assistant", content=prev_mailbody))
            else:
                msgs.append(Message(role="user", content=prev_mailbody))

        return msgs


    def create_arley_email_in_db_from_response(
        self,
        ollamamsgs: list[Message],
        new_text: str,
        ollama_response: dict,
        arley_email_address: str = settings.emailsettings.mailaddress,
        lang: Literal["de", "en"] = "de",
    ) -> ArleyEmailInDB | None:
        aeid: ArleyEmailInDB = ArleyEmailInDB()

        aeid.emailid = uuid.uuid4()
        aeid.ollamamsgs = ollamamsgs
        aeid.lang = lang
        aeid.processed = False
        aeid.processresult = Result.pending
        aeid.received = datetime.datetime.now(tz=DBObject.TIMEZONE)

        aeid.rootemailid = self.mailindb.rootemailid
        aeid.isrootemail = False
        aeid.frommail = arley_email_address
        aeid.tomail = self.mailindb.frommail
        aeid.toarley = False
        aeid.fromarley = True
        if self.mailindb.sequencenumber is not None:
            aeid.sequencenumber = self.mailindb.sequencenumber + 1
        aeid.envelopeemailid = utils.make_msgid(domain=Address(addr_spec=arley_email_address).domain)

        aeid.mailbody = new_text
        aeid.ollamaresponse = ollama_response

        aeid.subject = f"Re: {self.mailindb.subject}"
        aeid.sanitize_header_fields()

        detected_arleyid_in_subject: uuid.UUID | None = MyEmailMessage.get_arley_id(aeid.subject)
        self.logger.debug(f"{aeid.emailid=} {detected_arleyid_in_subject=}")

        # detected_arleyid_in_text: uuid.UUID | None = MyEmailMessage.get_arley_id(self.mailindb.mailbody)
        # self.logger.debug(f"{aeid.emailid=} {detected_arleyid_in_text=}")

        if not detected_arleyid_in_subject:
            aeid.subject = f"{aeid.subject} [arley-id {self.mailindb.rootemailid}]"

        res: DBObjectInsertUpdateDeleteResult | None = aeid.insertnew()
        if res is not None and res.exception_occured():
            ex: Exception|None = res.get_exception()
            if ex is not None:
                logger.error(Helper.get_exception_tb_as_string(ex))
            return None

        return aeid


    def process_request(self, streamed: bool = True) -> None:
        # streamed = True for long running responses from ollama, this makes waiting a bit easier, but logging could get a bit messy

        if self.mailindb.fromarley:
            raise RuntimeError("Mail is not FOR arley, but FROM arley...")

        temperature: float = 0.0

        self.mailindb.processresult = Result.working
        self.mailindb.save()
        self.logger.debug(Helper.get_pretty_dict_json_no_sort(self.mailindb))

        # str(mailbody), bool(fromarley)
        previous_texts: list[tuple[str, bool]] = []

        initial_topic: str|None = self.mailindb.subject
        initial_prompt: str|None = self.mailindb.mailbody

        # 1. die vorigen mails ziehen, wenn vorige da sein könnten
        if self.mailindb.sequencenumber is not None and self.mailindb.sequencenumber > 0:
            prev_sql: str = (
                f"select * from emails where rootemailid='{self.mailindb.rootemailid}' and sequencenumber<{self.mailindb.sequencenumber} order by sequencenumber asc"
            )
            previous: list[ArleyEmailInDB] = ArleyEmailInDB.get_list_from_sql(prev_sql)
            self.logger.debug(f"{prev_sql=}")

            initial_topic = previous[0].subject
            initial_prompt = previous[0].mailbody

            for prev in previous:
                if prev.mailbody is not None and prev.fromarley is not None:
                    previous_texts.append((prev.mailbody, prev.fromarley))

        self.logger.debug(f"{initial_topic=}")

        # in_reply_to: str = self.mailindb.envelopeemailid
        # references.write(in_reply_to)  # append email-id of the email replying to

        lang: Literal["de", "en"] | None = None
        ollama_response: dict | None = None
        lang_detect_content: dict | None = None


        assert initial_prompt is not None and initial_topic is not None
        lang_detect_text: str = (
            f"{initial_topic[:min(len(initial_prompt), 128)]}\n" f"{initial_prompt[:min(len(initial_prompt), 128)]}"
        )
        lang_detect_text = lang_detect_text.strip()

        ret: Literal['de', 'en']|Tuple[Literal['de', 'en'], dict, dict]|None = None
        try:
            ret = LanguageGuesser.guess_language(input_text=lang_detect_text, only_return_str=False,
                                                 ollama_host=OLLAMA_HOST, ollama_model=OLLAMA_GUESS_LANGUAGE_MODEL,
                                                 ollama_options=get_ollama_options(OLLAMA_GUESS_LANGUAGE_MODEL),
                                                 print_msgs=True, print_response=True, print_request=True, print_http_response=False,
                                                 print_http_request=False, max_retries=3)
        except Exception as ex:
            logger.exception(Helper.get_exception_tb_as_string(ex))

        if ret and isinstance(ret, tuple):
            lang, ollama_response, lang_detect_content = ret
            self.logger.debug(f"{lang=}")
            self.logger.debug(f"ollama_response:\n{Helper.get_pretty_dict_json_no_sort(ollama_response)}")
            self.logger.debug(f"lang_detect_content:\n{Helper.get_pretty_dict_json_no_sort(lang_detect_content)}")

        if lang is None:
            lang = "de"
            logger.error("DEFAULTING LANG to de")

        self.mailindb.lang = lang

        msgs: list[Message] = self.generate_msgs_for_ollama(
            lang=lang,
            previous_texts=previous_texts,
            initial_topic=initial_topic
        )

        assert self.mailindb.mailbody is not None
        prompt: str = self.mailindb.mailbody.strip()

        primer: list[Message] = OllamaEmailReply.generate_msgs_for_ollama(
            lang=lang, initial_topic=initial_topic, previous_texts=[]
        )

        refinelog: StringIO = StringIO()

        is_initial_request: bool = len(previous_texts) == 0

        _new_text_refined: str
        resp: dict

        call_start: float = perf_counter()
        _new_text_refined, resp = self.__class__.generate_response_from_ollama(
            is_initial_request=is_initial_request,
            msgs=msgs,
            initial_topic=initial_topic,
            prompt=prompt,
            lang=lang,  # type: ignore
            primer=primer,
            ollama_model=OLLAMA_MODEL,
            streamed=streamed,
            temperature=temperature,
            n_aug_results=ARLEY_AUG_NUM_DOCS,
            template_type=ARLEY_AUG_TEMPLATE_TYPE,
            unified_aug=ARLEY_AUG_UNIFIED,
            refine_per_item=ARLEY_AUG_PER_ITEM,
            aug_only_contracts=ARLEY_AUG_ONLY_CONTRACTS,
            aug_lang_filter=ARLEY_AUG_LANG_FILTER,
            first_request_unified=ARLEY_AUG_FIRST_REQUEST_UNIFIED,
            first_request_per_item=ARLEY_AUG_FIRST_REQUEST_PER_ITEM,
            first_request_aug_lang_filter=ARLEY_AUG_FIRST_REQUEST_AUG_LANG_FILTER,
            first_request_include_aug=ARLEY_AUG_FIRST_REQUEST_INCLUDE_AUG,
            first_request_template_type=ARLEY_AUG_FIRST_REQUEST_TEMPLATE_TYPE,
            first_request_n_aug_results=ARLEY_AUG_FIRST_REQUEST_N_AUG_RESULTS,
            first_request_aug_only_contracts=ARLEY_AUG_FIRST_REQUEST_AUG_ONLY_CONTRACTS,
            print_msgs=True,
            print_responses=True,
            refinelog=refinelog
        )
        call_end: float = perf_counter()

        self.logger.debug(f"OLLAMA CALL DONE in :: {(call_end-call_start):.2f}s")

        new_text_refined_wo_think_tag: str|None
        new_text_refined_think_tag: str|None
        new_text_refined_wo_think_tag, new_text_refined_think_tag = Helper.detach_think_tag(_new_text_refined)

        try:
            if REFINELOG_RECIPIENTS and len(REFINELOG_RECIPIENTS)>0:
                Helper.maillog(
                    text=refinelog.getvalue(),
                    subject=f"REFINELOG::{self.mailindb.rootemailid}",
                    from_mail=settings.emailsettings.mailuser,  # yikes.
                    mailrecipients_to=REFINELOG_RECIPIENTS,
                )
        except Exception as ex:
            logger.exception(ex)

        assert new_text_refined_wo_think_tag is not None
        new_answer: ArleyEmailInDB = self.create_arley_email_in_db_from_response(
            ollamamsgs=msgs,
            new_text=new_text_refined_wo_think_tag,  # _new_text_refined,
            ollama_response=resp,
            lang=lang  # type: ignore
        )

        if new_answer:
            self.logger.debug(f"NEW ANSWER CREATED IN DB -> setting mail[emailid={self.mailindb.emailid}] to processed")
            self.mailindb.processed = True
            self.mailindb.processresult = Result.processed  # type: ignore
            self.mailindb.save()
        else:
            # setting processresult to FAILED ?!
            self.logger.debug(f"NO NEW ANSWER CREATED IN DB for mail[emailid={self.mailindb.emailid}]")


    def handle_pending_mailout(self) -> None:
        if self.mailindb.toarley:
            raise RuntimeError("Mail is not FROM arley, but FOR arley...")

        self.mailindb.processresult = Result.working
        self.mailindb.save()
        self.logger.debug(Helper.get_pretty_dict_json_no_sort(self.mailindb))

        references: StringIO = StringIO()

        sql: str = (
            f"select * from emails where rootemailid='{self.mailindb.rootemailid}' and sequencenumber<{self.mailindb.sequencenumber} order by sequencenumber desc"
        )
        previous: list[ArleyEmailInDB] = ArleyEmailInDB.get_list_from_sql(sql)
        self.logger.debug(f"{sql} -> {len(previous)=}")

        # per logik gibt es eine email VOR dieser email
        answering_to_email_id: uuid.UUID|None = previous[0].emailid

        assert previous[0].envelopeemailid is not None
        in_reply_to: str = previous[0].envelopeemailid.strip()

        for prev in reversed(previous):
            references.write(f"{prev.envelopeemailid} ")

        references.write(in_reply_to)  # append email-id of the email replying to

        answerint_to_email_rawdata: ArleyRawEmailInDB|None = ArleyRawEmailInDB.get_one_from_sql(
            f"select * from rawemails where emailid='{answering_to_email_id}'"
        )

        assert answerint_to_email_rawdata is not None and previous[0].received is not None
        previous_text: str|None = answerint_to_email_rawdata.textmailbody
        previous_sender: str|None = previous[0].frommail
        previous_datum_dt: datetime.datetime = previous[0].received.astimezone(_timezone)
        previous_datum: str = previous_datum_dt.isoformat(timespec="seconds")

        assert previous_text is not None and previous_sender is not None
        sent_email: str = self.mailstuff(
            lang=self.mailindb.lang,  # type: ignore
            in_reply_to=in_reply_to,
            references=references.getvalue().rstrip(),
            previous_text=previous_text,
            previous_datum=previous_datum,
            previous_sender=previous_sender,
        )

        if sent_email:
            # NACH DEM MAILOUT:
            self.mailindb.processed = True
            self.mailindb.processresult = Result.processed
            self.mailindb.save()

            ima: IMAPAdapter = IMAPAdapter()
            ima.login()

            try:
                mails: list[tuple[int, Envelope]] = ima.list_mails(settings.emailsettings.folders.cur)
                for msgid, env in mails:
                    env_messageid: str = env.message_id.decode().strip()
                    self.logger.debug(f"{env_messageid=} {in_reply_to=} {self.mailindb.envelopeemailid=}")
                    if env_messageid == in_reply_to:
                        self.logger.debug("\tMATCHED")
                        ima.move_mail_to_done(
                            msgid=msgid,
                            folder_src=settings.emailsettings.folders.cur,
                            tofolder_sub=str(self.mailindb.rootemailid),
                        )

                ima.append(emaildata=sent_email, folder="Sent")  # double saving e-mail for convenience...

                # folder-existence was ensured somewhere else... if not... meh...
                ima.append(
                    emaildata=sent_email,
                    folder=f"{settings.emailsettings.folders.old}.{str(self.mailindb.rootemailid)}",
                )
            except Exception as ex:
                logger.exception(ex)
            finally:
                ima.logout()

    def mailstuff(
        self,
        lang: Literal["de", "en"],
        in_reply_to: str,
        references: str,
        previous_text: str,
        previous_sender: str,
        previous_datum: str,
        mailrecipients_cc: list[str] | None = None,
    ) -> str | None:
        fp: Path = Path(_templatedirpath, "mailtemplate.txt.jinja")
        self.logger.debug(f"{fp.absolute()=}")

        with open(fp) as file_:
            template = Environment(loader=FileSystemLoader(fp.parent), trim_blocks=True, lstrip_blocks=True).from_string(file_.read())

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
            returnpath=MailReport.EmailAddress.fromSTR(self.mailindb.frommail),
            replyto=MailReport.EmailAddress.fromSTR(self.mailindb.frommail),
            subject=self.mailindb.subject,
        )
        sendmail.tos = [MailReport.EmailAddress.fromSTR(k) for k in [self.mailindb.tomail]]

        if mailrecipients_cc is not None:
            sendmail.ccs = [MailReport.EmailAddress.fromSTR(k) for k in mailrecipients_cc]

        assert self.mailindb.ollamaresponse is not None
        values: dict = {
            "lang": lang,
            "schrieb_written_am": " schrieb am " if lang == "de" else " wrote on ",
            "previous_datum": previous_datum,
            "previous_sender": previous_sender,
            "previous_text": previous_text,
            "ollama_response_body": self.mailindb.mailbody,
            "arleyid": self.mailindb.rootemailid,
            "ollama_response": (
                self.mailindb.ollamaresponse
                if isinstance(self.mailindb.ollamaresponse, dict)
                else json.loads(self.mailindb.ollamaresponse)
            ),
        }

        del values["ollama_response"]["message"]

        mt_txt: str = template.render(values)
        self.logger.debug(mt_txt)

        additional_headers: dict[str, str] = {}
        additional_headers["In-Reply-To"] = in_reply_to
        additional_headers["References"] = references

        if not nosend:
            sent_mail: str = sendmail.send(
                txt=mt_txt, msgid=self.mailindb.envelopeemailid, additional_headers=additional_headers
            )
            return sent_mail

        return None

def _process_pending_mailouts(set_failed_mailout_to_failed_in_db: bool = False, newestfirst: bool = True) -> None:
    sqlout: str = (
        f"select * from emails where processresult='{Result.pending.value}' and fromarley order by received"
    )
    if not newestfirst:
        sqlout += " desc"  # dann die neuesten zuerst

    mailsout: list[ArleyEmailInDB] | None = ArleyEmailInDB.get_list_from_sql(sqlout)
    logger.debug(f"MAILOUT sqlout={sqlout} => {len(mailsout) if mailsout else 0}")

    if mailsout:
        for mail in mailsout:
            try:
                oer: OllamaEmailReply = OllamaEmailReply(mailindb=mail)
                oer.handle_pending_mailout()  # may raise Exception
            except Exception as ex:
                if set_failed_mailout_to_failed_in_db:
                    mail.processresult = Result.failed
                    mail.save()
                else:
                    raise ex



def _cleanup_from_previous_working_state() -> None:
    # this is a fresh start - as for now, the assumption is, that there is only ONE worker running the "queue"
    # -> just update WORKING-items in the DB to PENDING-itmes in the DB...

    sql_working_emails_from_last_run: str = (
        f"select * from emails where processresult='{Result.working.value}'"  # egal, ob VON oder AN arley
    )
    mails_working_from_previous_run: list[ArleyEmailInDB] | None = ArleyEmailInDB.get_list_from_sql(
        sql_working_emails_from_last_run)
    logger.debug(
        f"PREVIOUS_WORKING sql_working_emails_from_last_run={sql_working_emails_from_last_run} => {len(mails_working_from_previous_run) if mails_working_from_previous_run else 0}")

    if mails_working_from_previous_run:
        for mail in mails_working_from_previous_run:
            mail.processresult = Result.pending
            mail.save()
            logger.debug(f"setting email with emailid={mail.emailid} fromarley={mail.fromarley} to pending")

        _process_pending_mailouts(set_failed_mailout_to_failed_in_db=True, newestfirst=False)  # wenn die schon in der db sind, hat nur der mailout nicht funktioniert

def main(timeout_per_loop: int = 5, max_loop: int | None = None) -> Exception | None:
    try:
        _cleanup_from_previous_working_state()
    except Exception as ex:
        logger.exception(ex)
        # return ex  # not raising error here to allow other IN-requests to be processed

    try:
        loopcounter: int = 0
        while True:
            logger.debug(f"loopcounter={loopcounter}")

            loopcounter += 1
            sql: str = (
                f"select * from emails where processresult='{Result.pending.value}' and toarley order by received asc"
            )
            mails: list[ArleyEmailInDB] | None = ArleyEmailInDB.get_list_from_sql(sql)
            logger.debug(f"MAILIN sql={sql} => {len(mails) if mails else 0}")

            if mails:
                for mail in mails:
                    try:
                        oer: OllamaEmailReply = OllamaEmailReply(mailindb=mail)
                        oer.process_request()

                        try:
                            _process_pending_mailouts(set_failed_mailout_to_failed_in_db=False, newestfirst=True)
                        except Exception as exx:
                            logger.exception(exx)
                            # no error thrown/return from this block <- this block is just to increase responsiveness
                    except Exception as e:
                        logger.exception(e)
                        time.sleep(timeout_per_loop)
                        return e  # TODO HT 20240714 check!

            
            try:
                _process_pending_mailouts(set_failed_mailout_to_failed_in_db=False, newestfirst=True)
            except Exception as e:
                logger.exception(e)
                time.sleep(timeout_per_loop)
                return e

            if max_loop and loopcounter >= max_loop:
                break

            time.sleep(timeout_per_loop)
    except Exception as ex:
        logger.exception(ex)
        return ex

    return None
