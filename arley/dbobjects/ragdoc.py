import uuid
import datetime
from enum import StrEnum, auto
from os import stat_result
from pathlib import Path
from typing import Optional, Self
from uuid import UUID

import pytz
import ruamel.yaml

from arley import Helper
from arley.config import settings, is_in_cluster
from loguru import logger

from pydantic import BaseModel, Field
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str

_timezone: datetime.tzinfo = pytz.timezone(settings.timezone)

# THIS GOES INTO CHROMA DB -> NO sqlachemy stuff for it!

class FileFormatEnum(StrEnum):
    word = auto()
    excel = auto()
    markdown = auto()

class SentimentEnum(StrEnum):
    outstanding=auto()
    good=auto()
    positive=auto()
    neutral=auto()
    not_good=auto()
    wrong=auto()
    horrible=auto()

class LangEnum(StrEnum):
    # ISO 639 alpha-2
    de = auto()
    en = auto()

class DocTypeEnum(StrEnum):
    contract = auto()
    contract_excerpt = auto()
    general_clause = auto()
    template_clause = auto()
    variation = auto()
    contract_outline = auto()
    contract_summary = auto()

    targeted_by_prompts_lookup = auto()
    title_lookup = auto()
    categorization_lookup = auto()



class Categorization(BaseModel):
    tags: Optional[list[str]]
    relevance: int =  Field(..., gt=0, lt=500)
    sentiment: SentimentEnum

    # if these "questions/instructions are asked/given", the "system" shall consider this document
    targeted_by_prompts: Optional[list[str]] = Field(default=None)

    # some text which goes on and tells the system (in the language of the snippet what this is about)
    additional_notes: Optional[list[str]] = Field(default=None)

    titles: list[str]


class ArleyDocumentInformation(BaseModel):
    id: UUID = Field(default_factory=uuid.uuid4)
    parent_id: UUID | None = Field(default=None)
    md5: str | None = Field(default=None)

    title: str

    file_format: FileFormatEnum
    file_name: str
    full_path: Path

    date_created: datetime.datetime
    date_modified: datetime.datetime

    # the following three should only be given if this document is excerpted or otherwise decendant of that "parent" document
    parent_file_name: str | None = Field(default=None)
    parent_full_path: Path | None = Field(default=None)

    lang: LangEnum

    doctype: DocTypeEnum

    categorization: Categorization

    @classmethod
    def from_yaml_model_file(cls, yaml_file: Path) -> Optional[Self]:
        with open(yaml_file, "r") as yamlstream:
            return parse_yaml_raw_as(cls, yamlstream)


    @classmethod
    def from_xls_converted_yaml_file(cls,
                                     yaml_file: Path,
                                     title: str,
                                     doctype: str,
                                     parent_id: UUID | None = None,
                                     parent_file_name: str | None = None,
                                     parent_full_path: Path | None = None) -> Self | None:
        yaml_data: dict = None
        with open(yaml_file) as stream:
            try:
                yaml = ruamel.yaml.YAML(typ='safe', pure=True)
                yaml_data = yaml.load(stream)
            except ruamel.yaml.YAMLError as exc:
                logger.exception(exc)

        logger.debug(f"{type(yaml_data)=}")
        if not yaml_data:
            return None

        #datetime.datetime.now(tz=_timezone).isoformat(timespec="seconds"),

        # datetime.datetime.fromtimestamp(
        lstat: stat_result = yaml_file.lstat()

        return cls.from_xls_converted_yaml(
                yaml_data=yaml_data,
                doctype=doctype,
                title=title,
                file_name=yaml_file.name,
                full_path=yaml_file.resolve().absolute(),
                date_created=datetime.datetime.fromtimestamp(lstat.st_ctime, tz=_timezone),
                date_modified=datetime.datetime.fromtimestamp(lstat.st_mtime, tz=_timezone),
                md5=Helper.get_md5_for_file(yaml_file),
                parent_id=parent_id,
                parent_file_name=parent_file_name,
                parent_full_path=parent_full_path
        )


    @classmethod
    def from_xls_converted_yaml(cls,
                                yaml_data: dict,
                                doctype: str,
                                title: str,
                                file_name: str,
                                full_path: Path,
                                date_created: datetime.datetime,
                                date_modified: datetime.datetime,
                                md5: str,
                                parent_id: UUID | None = None,
                                parent_file_name: str | None = None,
                                parent_full_path: Path | None = None) -> Self:


        docid: UUID = uuid.uuid4()

        titles: str | list[str] = yaml_data["Titel"]
        if isinstance(titles, str):
            titles = [titles]

        tags: str | list[str] = yaml_data["Tags"]
        if isinstance(tags, str):
            tags = [tags]

        targeted_by_prompts: str | list[str] = yaml_data["Targeted By Prompts"]
        if isinstance(targeted_by_prompts, str):
            targeted_by_prompts = [targeted_by_prompts]

        additional_notes: None | str | list[str] = yaml_data["Additional Notes"]
        if additional_notes and isinstance(additional_notes, str):
            additional_notes = [additional_notes]

        sprache: None | str = yaml_data.get("Sprache")
        if sprache:
            sprache = sprache.lower()
        else:
            sprache = LangEnum.de.value

        categorization: Categorization = Categorization(
            tags=tags,  # only reference -> NO copy!
            relevance=yaml_data["Relevanz"],
            sentiment=SentimentEnum[yaml_data["Sentiment"].lower()],
            targeted_by_prompts=targeted_by_prompts,
            additional_notes=additional_notes,
            titles=titles
        )


        ret: ArleyDocumentInformation = ArleyDocumentInformation(
            id=docid,
            doctype=DocTypeEnum[doctype.lower()],
            parent_id=parent_id,
            file_format=FileFormatEnum[yaml_data["Format"].lower()],
            lang=LangEnum[sprache],
            md5=md5,
            title=title,  # for now, just take the "filename" since the "title" is completely rubbish as a list
            parent_file_name=parent_file_name,
            parent_full_path=parent_full_path,
            file_name=file_name,
            full_path=full_path,
            date_created=date_created,
            date_modified=date_modified,
            categorization=categorization
        )

        return ret

