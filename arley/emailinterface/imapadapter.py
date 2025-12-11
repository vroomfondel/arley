import email
import time
from collections import defaultdict
from datetime import datetime
from email.header import decode_header
from email.message import EmailMessage, Message
from email.utils import parseaddr
from typing import Any
from uuid import UUID

from imapclient import \
    IMAPClient  # imapclient is more recent than imaplib2 -> https://github.com/mjs/imapclient/
from imapclient.response_types import Address, BodyData, Envelope, SearchIds
from loguru import logger
from mailparser_reply import EmailReply

from arley import Helper
from arley.config import (ARLEY_IMAPLOOP_MAX_IDLE_LOOPS,
                          ARLEY_IMAPLOOP_MAX_IDLE_UNSUCCESS_IN_SEQUENCE,
                          ARLEY_IMAPLOOP_TIMEOUT_PER_IDLE_LOOP, settings)
from arley.dbobjects.emailindb import ArleyEmailInDB, ArleyRawEmailInDB
from arley.emailinterface.myemailmessage import MyEmailMessage

# import imaplib2


LOGME_VERBOSE: bool = False
ONLY_MAILS_FROM_PRIVILEGED_SENDER: bool = False


class IMAPAdapter:
    logger = logger.bind(classname=__qualname__)

    def __init__(self) -> None:
        self.imapclient: IMAPClient = IMAPClient(
            settings.emailsettings.imapserver, port=settings.emailsettings.imapport, use_uid=True, ssl=False
        )

    def logout(self) -> None:
        self.imapclient.logout()
        # setting self.imapclient to None ?!

    def login(self) -> None:
        self.imapclient.starttls()
        resp: list = self.imapclient.login(settings.emailsettings.mailuser, settings.emailsettings.mailpassword)

        if LOGME_VERBOSE:
            for r in resp:
                self.logger.debug(f"RESP: {type(r)=} {r=}")

    def append(self, emaildata: str, folder: str = "INBOX", msg_time: Any = None, flags: tuple = tuple()) -> list[str]:
        return self.imapclient.append(folder=folder, msg=emaildata, msg_time=msg_time, flags=flags)

    def list_folders(self, folder: str = "") -> list[str]:
        # select_info: dict[bytes, tuple[bytes, ...] | tuple | bool | int] = self.imapclient.select_folder(
        #     folder, readonly=True
        # )
        # if LOGME_VERBOSE:
        #     self.logger.debug(select_info)
        #     self.logger.debug(f"{select_info[b'EXISTS']} messages in {folder}")

        ret: list[str] = []
        # (flags | delim | name)
        folders: list[tuple[None | int | bytes, None | int | bytes, str | None | bytes]] = self.imapclient.list_folders(
            directory=folder
        )
        for fol in folders:
            flags, delim, name = fol
            if LOGME_VERBOSE:
                self.logger.debug(f"FOLDER: {type(flags)=} {flags=}\t{type(delim)=} {delim=}\t{type(name)=} {name=}")
            ret.append(str(name))

        return ret

    # imapclient.list_sub_folders()

    def ensure_folders(self, need: set[str], parent_folder: str = "") -> None:
        """
        woohoo

        :param need: set of absolute(!!!) folder-names to ensure
        :param parent_folder: used as filter to speed up things when detecting if needed folders missing
        :return: None
        """
        existing_folders: list[str] = self.list_folders(folder=parent_folder)
        for existing_folder in existing_folders:
            # flags, delim, name = folder
            # self.logger.debug(f"FOLDER: {type(flags)=} {flags=}\t{type(delim)=} {delim=}\t{type(name)=} {name=}")
            if existing_folder in need:
                need.remove(existing_folder)

        for name in need:
            assert parent_folder == "" or name.startswith(f"{parent_folder}.")

            self.logger.debug(f"FOLDER STILL NEEDED: {name}")
            # resp = self.imapclient.create_folder(f"{folder}.{name}")

            resp = self.imapclient.create_folder(name)
            self.logger.debug(f"{type(resp)=}")
            self.logger.debug(resp)
            resp_str: str = resp.decode("utf-8")
            ok: bool = resp_str.find("Create completed") == 0

            if ok:
                self.logger.debug(f"Creation of folder {name} succeeded.")
                resp = self.imapclient.subscribe_folder(name)
                self.logger.debug(f"{type(resp)=}")
                self.logger.debug(resp)
            else:
                raise RuntimeError(f"IMAP FOLDER CREATION FAILED {name} => {resp_str=}")

    def ensure_base_folders(self, parent_folder: str = "") -> None:
        need: set = set()
        for name, value in settings.emailsettings.folders.model_fields.items():
            actual_name: str = getattr(settings.emailsettings.folders, name)
            self.logger.debug(f"{name=} {value=} {actual_name=}")
            need.add(actual_name)

        self.ensure_folders(need=need, parent_folder=parent_folder)

    def list_mails(self, folder: str = "INBOX") -> list[tuple[int, Envelope]]:
        select_info: dict[bytes, tuple[bytes, ...] | tuple | bool | int] = self.imapclient.select_folder(
            folder, readonly=True
        )
        if LOGME_VERBOSE:
            self.logger.debug(select_info)
        # {b'PERMANENTFLAGS': (), b'FLAGS': (b'\\Answered', b'\\Flagged', b'\\Deleted', b'\\Seen', b'\\Draft', b'NonJunk'), b'EXISTS': 7, b'RECENT': 0, b'UNSEEN': [b'3'], b'UIDVALIDITY': 1720445093, b'UIDNEXT': 19, b'HIGHESTMODSEQ': 36, b'READ-ONLY': [b'']}

        ret: list[tuple[int, Envelope]] = []
        searchids: SearchIds = self.imapclient.search()  # serach-term möglich
        if LOGME_VERBOSE:
            self.logger.debug(f"{type(searchids)=}")
            self.logger.debug(searchids)

        for uid, message_data in self.imapclient.fetch(searchids, "ENVELOPE").items():
            if LOGME_VERBOSE:
                self.logger.debug(f"{type(message_data)=}")
                self.logger.debug(message_data)

            envelope: Envelope = message_data[b"ENVELOPE"]
            if LOGME_VERBOSE:
                self.logger.debug(f"{uid=}")

            if LOGME_VERBOSE:
                self.logger.debug(f"{type(envelope)=}")
                self.logger.debug(envelope)

                self.logger.debug(f"{envelope.from_=}")
                self.logger.debug(f"{envelope.sender}")

            # print('ID #%d: "%s" received %s' % (msgid, envelope.subject.decode(), envelope.date))

            ret.append((uid, envelope))

        # if len(ret) == 0:
        #     return None

        return ret

    def get_message(self, msgid: int, folder: str = "INBOX") -> MyEmailMessage | None:
        select_info: dict[bytes, tuple[bytes, ...] | tuple | bool | int] = self.imapclient.select_folder(
            folder, readonly=True
        )
        if LOGME_VERBOSE:
            self.logger.debug(select_info)

        ret: MyEmailMessage | None = None

        fetched: (
            dict | defaultdict[int, dict[bytes, datetime | int | BodyData | Envelope | None | bytes | tuple[Any, ...]]]
        ) = self.imapclient.fetch(msgid, "RFC822")
        if LOGME_VERBOSE:
            self.logger.debug(f"{type(fetched)=}")

        message_data = fetched[msgid]
        if LOGME_VERBOSE:
            self.logger.debug(f"{type(message_data)=}")
            self.logger.debug(f"{message_data=}")

        email_message: EmailMessage = email.message_from_bytes(message_data[b"RFC822"])  # type: ignore

        subject: str | None = MyEmailMessage.decode_my_header(email_message.get("Subject"))

        from_email: str | None = parseaddr(MyEmailMessage.decode_my_header(email_message.get("From")))[1]  # type: ignore
        to_email: str | None = parseaddr(MyEmailMessage.decode_my_header(email_message.get("To")))[1]  # type: ignore # Delivered-to
        envelope_message_id: str | None = MyEmailMessage.decode_my_header(email_message.get("Message-ID")).rstrip().lstrip()  # type: ignore
        if LOGME_VERBOSE:
            self.logger.debug(f"{msgid=} {envelope_message_id=}\t{from_email=} {to_email=} {subject=}")

        # only debug-info here...
        in_reply_to: str | None = MyEmailMessage.decode_my_header(email_message.get("In-Reply-To"))
        if in_reply_to:
            in_reply_to = in_reply_to.rstrip().lstrip()
        if LOGME_VERBOSE:
            self.logger.debug(f"{in_reply_to=}")

        references: str | None = MyEmailMessage.decode_my_header(email_message.get("References"))
        if references:
            references = references.rstrip().lstrip()
        if LOGME_VERBOSE:
            self.logger.debug(f"{references=}")

        assert (
            from_email is not None and to_email is not None and subject is not None and envelope_message_id is not None
        )
        ret = MyEmailMessage(
            msgid=msgid,
            email_message=email_message,
            from_email=from_email,
            to_email=to_email,
            subject=subject,
            envelope_message_id=envelope_message_id,
        )

        return ret

    # c. fetch([3293, 3230], ['INTERNALDATE', 'FLAGS'])
    # for msgid, data in imapclient.fetch(messages, ['ENVELOPE']).items():
    #     envelope = data[b'ENVELOPE']
    #     print('ID #%d: "%s" received %s' % (msgid, envelope.subject.decode(), envelope.date))

    # imapclient_idle(imapclient=imapclient, folder='INBOX')
    # imapclient_get_unseen(imapclient=imapclient, folder='INBOX')
    # imapclient.logout()

    def idle(self, folder: str = "INBOX", timeoutperloop: int = 10, maxloops: int | None = None) -> bool:
        select_info: dict[bytes, tuple[bytes, ...] | tuple | bool | int] = self.imapclient.select_folder(folder)
        if LOGME_VERBOSE:
            self.logger.debug(select_info)

        # Start IDLE mode
        idle_resp = self.imapclient.idle()
        self.logger.debug(f"{type(idle_resp)=} {idle_resp=} {maxloops=} {timeoutperloop=}")
        self.logger.debug(f"{maxloops=} {timeoutperloop=}")
        self.logger.debug("Connection is now in IDLE mode, send yourself an email or quit with ^c")

        ret: bool = False

        ki: KeyboardInterrupt | None = None
        loopcount: int = 0
        while True:
            loopcount += 1
            try:
                responses = self.imapclient.idle_check(timeout=timeoutperloop)
                self.logger.debug(
                    f"#{loopcount:>3} Server sent ({type(responses)=}): {responses if responses else "<nothing>"}"
                )
                if responses:
                    ret = True
                    break
            except KeyboardInterrupt as _ki:
                ki = _ki
                break

            if maxloops is not None and loopcount >= maxloops:
                self.logger.debug(f"breaking since {maxloops=}, {loopcount=} and {loopcount}>={maxloops}")
                break

        self.imapclient.idle_done()
        self.logger.debug(f"IDLE mode done :: {ret=}")

        if ki:
            self.logger.debug(f"raising KeybordInterrupt {ki=}")
            raise ki

        return ret

    def get_by_flag(self, folder: str = "INBOX", flag: str = "UNSEEN") -> list[Message] | None:
        select_info = self.imapclient.select_folder(folder, readonly=True)
        if LOGME_VERBOSE:
            self.logger.debug(select_info)

        ret: list[Message] = []
        messages: SearchIds = self.imapclient.search("UNSEEN")
        for uid, message_data in self.imapclient.fetch(messages, "RFC822").items():
            # check: https://docs.python.org/3/library/email.message.html
            email_message: Message = email.message_from_bytes(message_data[b"RFC822"])

            # email_message.print_info()

            self.logger.debug(email_message.get("Sender"))

            self.logger.debug(
                uid,
                MyEmailMessage.decode_my_header(email_message.get("From")),
                MyEmailMessage.decode_my_header(email_message.get("Subject")),
            )
            ret.append(email_message)

        if len(ret) == 0:
            return None

        return ret

    def imapclient_get_unseen(self, folder: str = "INBOX") -> list[Message] | None:
        return self.get_by_flag(folder=folder, flag="UNSEEN")

    def imapclient_get_recent(self, folder: str = "INBOX") -> list[Message] | None:
        return self.get_by_flag(folder=folder, flag="RECENT")

    def move_mail(self, msgid: int, folder_src: str = "INBOX", folder_dst: str = "Trash") -> None:
        select_info = self.imapclient.select_folder(folder_src)
        if LOGME_VERBOSE:
            self.logger.debug(select_info)

        resp = self.imapclient.move([msgid], folder_dst)

        self.logger.debug(f"{type(resp)=}")
        self.logger.debug(f"{resp=}")

        return resp

    def verify_sender(self, msgid: int, envelope: Envelope) -> str | None:
        sender_tpl = envelope.sender
        sender: Address = sender_tpl[0]

        replyto_tpl = envelope.reply_to
        reply_to: Address = replyto_tpl[0]

        from_tpl = envelope.from_
        from_: Address = from_tpl[0]

        self.logger.debug(f"{msgid=} {type(sender)=} {sender=} {type(reply_to)=} {reply_to=} {type(from_)=} {from_=}")

        domain: str = sender.host.decode()
        mbox: str = sender.mailbox.decode()

        if domain not in settings.emailsettings.alloweddomains:
            self.logger.debug(f"DOMAIN NOT ALLOWED: {domain}")
            return None

        return f"{mbox}@{domain}"

    def move_mail_to_rejected(self, msgid: int, folder_src: str = "INBOX") -> None:
        self.move_mail(msgid, folder_src=folder_src, folder_dst=settings.emailsettings.folders.rej)

    def move_mail_to_done(self, msgid: int, folder_src: str = "INBOX", tofolder_sub: str | None = None) -> None:
        dst: str = settings.emailsettings.folders.old
        if tofolder_sub:
            dst = f"{dst}.{tofolder_sub}"

        self.move_mail(msgid, folder_src=folder_src, folder_dst=dst)

    def move_mail_to_working(self, msgid: int, folder_src: str = "INBOX") -> None:
        self.move_mail(msgid, folder_src=folder_src, folder_dst=settings.emailsettings.folders.cur)

    def move_mail_to_err(self, msgid: int, folder_src: str = "INBOX") -> None:
        self.move_mail(msgid, folder_src=folder_src, folder_dst=settings.emailsettings.folders.err)

    def process_email(self, sender: str, mymail: MyEmailMessage) -> None:
        mymail.print_info()

        # 1. move email to "working"-directory
        self.move_mail_to_working(msgid=mymail.msgid)

        # TODO HT 20240714 work on email here
        # 2. check if new email (no reference) or old email (reference/arley-id found in text) [arley-id {{ ARLEYID }}]
        arley_id_from_subject_and_text: UUID | None = mymail.get_arley_ids_from_subject_and_text()

        previous_emails: list[ArleyEmailInDB] | None = None

        if arley_id_from_subject_and_text:
            sql: str = (
                f"select * from emails where emailid='{str(arley_id_from_subject_and_text)}' or rootemailid='{str(arley_id_from_subject_and_text)}' order by sequencenumber desc"
            )
            previous_emails = ArleyEmailInDB.get_list_from_sql(sql)

            self.logger.debug(f"{sql} -> {len(previous_emails)}")

        rootemailid: UUID | None = None
        if previous_emails:
            r: ArleyEmailInDB
            for r in previous_emails:
                logger.debug(Helper.get_pretty_dict_json_no_sort(r))
                if r.rootemailid:
                    logger.debug(f"setting rootemailid to: {r.rootemailid=}")
                    rootemailid = r.rootemailid

        # 3. insert appropriately into db
        aeid: ArleyEmailInDB | None = ArleyEmailInDB.insert_from_myemail(
            myemail=mymail, rootemailid=rootemailid, arley_email=settings.emailsettings.mailaddress, lang="de"
        )
        if not aeid:
            # TODO HT 20240714 handle exception properly
            ex: Exception = Exception("could not create email-entry in db")
            self.logger.error(ex)

        assert aeid is not None
        areid: ArleyRawEmailInDB | None = ArleyRawEmailInDB.insert_rawemail_from_myemail(
            myemail=mymail, arleyemailindb=aeid
        )
        if not areid:
            # TODO HT 20240714 handle exception properly
            ex2: Exception = Exception("could not create email-entry in db")
            self.logger.error(ex2)

        # 4. if needed, create arley-id folder in "old"
        need: set[str] = {f"{settings.emailsettings.folders.old}.{str(aeid.rootemailid)}"}
        # would fail if settings.emailsettings.folders.old is none or ""
        self.ensure_folders(need=need, parent_folder=settings.emailsettings.folders.old)

        # 5. hand over db-object to ollama-mailreply
        # ollama-handlers runs in other pod

        # reps: list[EmailReply] = mymail.get_all_replies()
        # for i in reps:
        #     self.logger.debug(f"{'*' * 20}")
        #     self.logger.debug(f"{type(i)=}")
        #     self.logger.debug(f"{i.body=}")
        #     self.logger.debug(f"{i.signatures=}")
        #     self.logger.debug(f"{i.disclaimers=}")
        #     self.logger.debug(f"{i=}")
        #
        # self.logger.debug(f"{'*' * 20}")
        # self.logger.debug(f"{'*' * 20}")
        #
        # latest_rep: EmailReply = mymail.get_latest_reply()
        # self.logger.debug(f"{type(latest_rep)=}")
        # self.logger.debug(f"{latest_rep=}")

    def work_inbox(self, timeout_if_exception_occured: int = 10) -> None:
        mails: list[tuple[int, Envelope]] = self.list_mails()

        for mt in mails:
            try:
                msgid: int = mt[0]
                env: Envelope = mt[1]

                sender: str | None = self.verify_sender(msgid=msgid, envelope=env)

                if not sender:
                    self.move_mail_to_rejected(msgid=msgid)
                    continue

                # could also add an array/set here for privileged senders
                if sender != settings.emailsettings.privileged_sender and ONLY_MAILS_FROM_PRIVILEGED_SENDER:
                    # skip
                    continue

                mymail: MyEmailMessage | None = self.get_message(msgid)
                if mymail is not None:
                    self.process_email(sender, mymail)
            except Exception as ex:
                self.logger.exception(ex)
                self.logger.info(f"sleeping {timeout_if_exception_occured}s after error occured")
                time.sleep(timeout_if_exception_occured)


# import email
# msg = email.message_from_bytes(myBytes)


def main(
    max_idle_unsuccess_in_sequence: int | None = None, max_idle_loops: int | None = None, timeout_per_idle_loop: int = 5
) -> Exception | None:
    """
    set max_idle_unsuccess_in_sequence to None to not exit (and thus have k8s restart the pod)
    set max_idle_loops to None to have it idle-loop for eternity (or received a signal while idleing)
    set timeout_per_idle_loop to None to not wait-timeout in each idle-loop
    """
    ima: IMAPAdapter = IMAPAdapter()
    ima.login()

    unsuccess_idle_count: int = 0
    try:
        ima.ensure_base_folders()

        while True:
            ima.work_inbox()  # work on everything that is already there
            # ima.list_folders()
            triggered_in_idle: bool = ima.idle(maxloops=max_idle_loops, timeoutperloop=timeout_per_idle_loop)

            if triggered_in_idle:
                # reset unsuccess-idle-count
                unsuccess_idle_count = 0
            else:
                unsuccess_idle_count += 1

            if max_idle_unsuccess_in_sequence and unsuccess_idle_count >= max_idle_unsuccess_in_sequence:
                logger.debug(f"{unsuccess_idle_count >= max_idle_unsuccess_in_sequence=}")
                break

    except Exception as ex:
        logger.exception(ex)
        return ex
    finally:
        ima.logout()

    return None


def testmode() -> None:
    ima: IMAPAdapter = IMAPAdapter()
    ima.login()

    logger.debug(Helper.get_pretty_dict_json_no_sort(ima.list_folders(folder="WORKED")))
    logger.debug("*" * 50)
    logger.debug(Helper.get_pretty_dict_json_no_sort(ima.list_folders(folder="INBOX")))

    ima.ensure_base_folders()

    ima.logout()


if __name__ == "__main__":
    res: Exception | None = main(
        max_idle_unsuccess_in_sequence=ARLEY_IMAPLOOP_MAX_IDLE_UNSUCCESS_IN_SEQUENCE,
        max_idle_loops=ARLEY_IMAPLOOP_MAX_IDLE_LOOPS,
        timeout_per_idle_loop=ARLEY_IMAPLOOP_TIMEOUT_PER_IDLE_LOOP,
    )

    if res:
        exit(123)

    exit(0)
