import datetime
import difflib
import textwrap
from io import StringIO
from pathlib import Path
from typing import Literal, Optional, TextIO, Tuple

import chromadb
import pytz
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from arley import Helper
from arley.config import (ARLEY_AUG_LANG_FILTER, ARLEY_AUG_NUM_DOCS,
                          ARLEY_AUG_ONLY_CONTRACTS, ARLEY_AUG_PER_ITEM,
                          ARLEY_AUG_TEMPLATE_TYPE, ARLEY_AUG_UNIFIED,
                          OLLAMA_MODEL, TEMPLATEDIRPATH, TemplateType,
                          settings)
from arley.llm.ollama_adapter import Message, ask_ollama_chat
from arley.vectorstore.chroma_adapter import ChromaDBConnection

_timezone: datetime.tzinfo = pytz.timezone(settings.timezone)

from chromadb.api.models.Collection import Collection as ChromaCollection


class OllamaChromaDBRAG:
    logger = logger.bind(classname=__qualname__)

    def __init__(self, cdbcollection: ChromaCollection):
        self.cdbcollection = cdbcollection

    def get_refine_contexts(
        self,
        prompt: str,
        lang: Literal["de", "en"],
        n_aug_results: int = ARLEY_AUG_NUM_DOCS,
        template_type: TemplateType = ARLEY_AUG_TEMPLATE_TYPE,
        unified_aug: bool = ARLEY_AUG_UNIFIED,
        refine_per_item: bool = ARLEY_AUG_PER_ITEM,
        aug_only_contracts: bool = ARLEY_AUG_ONLY_CONTRACTS,
        aug_lang_filter: bool = ARLEY_AUG_LANG_FILTER,
        is_initial_aug_request: bool = False,
        refinelog: StringIO | TextIO | None = None,
    ) -> list[Tuple[str, str]]:

        logger = self.__class__.logger

        context_aug: dict[str, list[dict]] = ChromaDBConnection.get_context_augmentations(
            initial_topic=None,  # initial_topic,
            prompt=prompt,
            lang=lang if aug_lang_filter else None,
            cdbcollection=self.cdbcollection,
            n_results=n_aug_results,
            only_contracts=aug_only_contracts,
        )
        logger.debug(f"CONTEXT_AUG:\n{Helper.get_pretty_dict_json_no_sort(context_aug)}")
        if refinelog:
            refinelog.write(
                f"\n\n\n\n################\nCONTEXT_AUG:\n{Helper.get_pretty_dict_json_no_sort(context_aug)}"
            )

        refines: list[Tuple[str, str]]
        if unified_aug:
            context_str: str = self.get_all_in_context_str(
                lang=lang,
                augments=context_aug,
                template_type=template_type,
                is_initial_aug_request=is_initial_aug_request,
            )
            refines = [("ALL", context_str)]
        else:
            refines = self.get_context_refines(
                lang=lang,
                augments=context_aug,
                template_type=template_type,
                per_item=refine_per_item,
                is_initial_aug_request=is_initial_aug_request,
            )

        return refines

    def refine_response(
        self,
        initial_topic: str,
        prompt: str,
        existing_response: str,
        lang: Literal["en", "de"],
        primer: list,
        ollama_model: str = OLLAMA_MODEL,
        streamed: bool = False,
        temperature: float = 0.1,
        n_aug_results: int = ARLEY_AUG_NUM_DOCS,
        template_type: TemplateType = ARLEY_AUG_TEMPLATE_TYPE,
        unified_aug: bool = ARLEY_AUG_UNIFIED,
        refine_per_item: bool = ARLEY_AUG_PER_ITEM,
        aug_only_contracts: bool = ARLEY_AUG_ONLY_CONTRACTS,
        aug_lang_filter: bool = ARLEY_AUG_LANG_FILTER,
        is_initial_aug_request: bool = False,
        is_history_mode: bool = False,
        refinelog: StringIO | TextIO | None = None,
        print_msgs: bool = False,
        print_response: bool = False,
    ) -> str:

        logger = self.__class__.logger

        if n_aug_results <= 0:
            return existing_response

        refines: list[Tuple[str, str]] = self.get_refine_contexts(
            prompt=prompt,
            lang=lang,
            n_aug_results=n_aug_results,
            template_type=template_type,
            refine_per_item=refine_per_item,
            aug_only_contracts=aug_only_contracts,
            aug_lang_filter=aug_lang_filter,
            is_initial_aug_request=is_initial_aug_request,
            unified_aug=unified_aug,
            refinelog=refinelog,
        )

        if len(refines) == 0:
            return existing_response

        notelines: list[str | None] = []

        refined_response_txt: str = existing_response
        refined_response_txt_wo_think_tag: str
        refined_response_txt_think_tag: str | None

        refined_response_txt_wo_think_tag, refined_response_txt_think_tag = Helper.detach_think_tag(existing_response)  # type: ignore

        for refine_type, merefine in refines:
            refined_prompt: str = self.get_refine_prompt(
                original_prompt=prompt,
                existing_response=refined_response_txt_wo_think_tag,
                context=merefine,
                ollama_model=ollama_model,
                lang=lang,
                template_type=template_type,
                is_initial_aug_request=is_initial_aug_request,
            )

            logger.debug(f"refined prompt ({refine_type=}):\n{textwrap.indent(refined_prompt, "  Q  ")}")

            if refinelog:
                refinelog.write(f"\n\n\n\n################\nREFINED PROMPT ({refine_type=}):\n{refined_prompt}")

            # wenn ich chatmode bin, muß ich die letzte nachricht austauschen -> sonst wird die nicht besser

            resp_refined: dict = ask_ollama_chat(  # type: ignore
                system_prompt=None,  # need to provide msg_history then
                prompt=refined_prompt,
                msg_history=primer,
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
                print_response=print_response,
                print_options=False,
                max_tries_ollama_done_response=21,
                streamed=streamed,
                print_chunks_when_streamed=False,
            )

            refined_response_txt_b: str = refined_response_txt
            refined_response_txt_wo_think_tag_b: str = refined_response_txt_wo_think_tag

            # TODO !!!

            refined_response_txt = resp_refined["message"]["content"]
            refined_response_txt_wo_think_tag, refined_response_txt_think_tag = Helper.detach_think_tag(refined_response_txt)  # type: ignore

            noteline: Optional[str]
            refined_response_txt_wo_think_tag, noteline = Helper.detach_NOTE_line(refined_response_txt_wo_think_tag)
            notelines.append(noteline)

            if is_history_mode:
                primer[-1]["content"] = refined_response_txt_wo_think_tag

            logger.debug(f"refined_response_txt_think_tag ({refine_type=}):\n{textwrap.indent(refined_response_txt_think_tag, "  R  ")}")  # type: ignore
            logger.debug(
                f"refined_response_txt ({refine_type=}):\n{textwrap.indent(refined_response_txt_wo_think_tag, "  R  ")}"
            )
            logger.debug(f"refined_response_txt ({refine_type=}) NOTELINE:\n{textwrap.indent(f"{noteline}", "  R  ")}")

            if refinelog and not refine_type == "ALL":
                # f"refined_response_txt ({refine_type=}):\n{textwrap.indent(refined_response_txt, "  R  ")}"
                refinelog.write(
                    f"\nrefined_response_txt ({refine_type=}) NOTE:\n{textwrap.indent(f"{noteline}", "  R  ")}\n"
                )

                diff_str: StringIO = StringIO()
                for l in difflib.unified_diff(
                    refined_response_txt_wo_think_tag_b,
                    refined_response_txt_wo_think_tag,
                    fromfile=f"BEFORE_{refine_type}",
                    tofile=f"AFTER_{refine_type}",
                ):
                    diff_str.write(l)

                refinelog.write(f"\n\n################\nDIFF_{refine_type}:\n{diff_str.getvalue()}")

        if refinelog:
            refinelog.write(
                f"\n\n\n\n################\nEXISTING RESPONSE TEXT:\n{textwrap.indent(existing_response, "  I  ")}"
            )
            refinelog.write(
                f"\n\n\n\n################\nREFINED RESPONSE TEXT:\n{textwrap.indent(refined_response_txt, "  R  ")}"
            )  # including think-tag!

            notelines_str: StringIO = StringIO()
            for i in range(0, len(notelines)):
                if i > 1:
                    notelines_str.write("\n")
                notelines_str.write(textwrap.indent(f"[{i+1}] {notelines[i]}\n", "  RN "))

            refinelog.write(f"\nREFINED RESPONSE TEXT NOTELINES:\n{notelines_str.getvalue()}")

            diff_str = StringIO()
            for l in difflib.unified_diff(
                existing_response, refined_response_txt_wo_think_tag, fromfile="B", tofile="A"
            ):
                diff_str.write(l)

            refinelog.write(f"\n\n\n\n################\nDIFF:\n{diff_str.getvalue()}")

        return refined_response_txt_wo_think_tag

    def get_refine_prompt(
        self,
        original_prompt: str,
        existing_response: str | None,
        context: str,
        ollama_model: str | None = None,
        lang: Literal["en", "de"] = "en",
        template_type: TemplateType = TemplateType.plain,
        is_initial_aug_request: bool = False,
    ) -> str:

        # https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/

        tname: str = f"refine_prompt_{lang}_{template_type.value}.jinja"
        if is_initial_aug_request:
            tname = f"aug_initial_prompt_{lang}_{template_type.value}.jinja"

        fp: Path = Path(TEMPLATEDIRPATH, template_type.value)
        fp = Path(fp, tname)

        with open(fp) as file_:
            template = Environment(
                loader=FileSystemLoader(fp.parent), trim_blocks=True, lstrip_blocks=True
            ).from_string(file_.read())

        values: dict = {
            "lang": lang,
            "original_prompt": original_prompt,
            "existing_response": existing_response,
            "context": context,
            "ollama_model": ollama_model,
            "is_initial_request": is_initial_aug_request,
            "template_type": template_type.value,
        }

        ret: str = template.render(values)
        return ret

    def get_context_refines(
        self,
        lang: Literal["de", "en"],
        augments: dict[str, list[dict]],
        ollama_model: str | None = None,
        template_type: TemplateType = TemplateType.plain,
        per_item: bool = True,
        is_initial_aug_request: bool = False,
        prompt: str | None = None,
    ) -> list[Tuple[str, str]]:

        ret: list[Tuple[str, str]] = []

        tname: str = f"get_context_{lang}_{template_type.value}.jinja"
        if is_initial_aug_request:
            tname = f"aug_initial_prompt_context_{lang}_{template_type.value}.jinja"

        fp: Path = Path(TEMPLATEDIRPATH, template_type.value)
        fp = Path(fp, tname)

        with open(fp) as file_:
            template = Environment(
                loader=FileSystemLoader(fp.parent), trim_blocks=True, lstrip_blocks=True
            ).from_string(file_.read())

        for refine_type in [
            "contract_outline",
            "contract_summary",
            "contract_excerpt",
            "contract",
        ]:  # context_aug.keys():
            merefine: list[dict] | None = augments.get(refine_type)
            if not merefine:
                logger.debug(f"SKIP: {refine_type}")
                continue

            logger.debug(f"FOUND: {refine_type}")

            values: dict = {
                "lang": lang,
                "refine_type": refine_type,
                "merefine": merefine,
                "ollama_model": ollama_model,
                "prompt": prompt,
                "template_type": template_type.value,
            }

            if per_item:
                for merefine_one in merefine:
                    values["merefine"] = [merefine_one]

                    ret.append((refine_type, template.render(values)))
            else:
                ret.append((refine_type, template.render(values)))

        return ret

    def get_all_in_context_str(
        self,
        lang: Literal["de", "en"],
        augments: dict[str, list[dict]],
        ollama_model: str | None = None,
        template_type: TemplateType = TemplateType.plain,
        is_initial_aug_request: bool = False,
    ) -> str:

        ret: StringIO = StringIO()

        refines: list[Tuple[str, str]] = self.get_context_refines(
            lang=lang,
            augments=augments,
            template_type=template_type,
            ollama_model=ollama_model,
            per_item=False,
            is_initial_aug_request=is_initial_aug_request,
        )

        for refine_type, merefine in refines:
            ret.write(merefine.strip())
            ret.write("\n\n")

        return ret.getvalue().strip()
