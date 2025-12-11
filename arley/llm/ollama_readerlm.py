
# os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"


import json

from pathlib import Path

from arley.llm.ollama_adapter import ask_ollama_chat

from loguru import logger



def main(fp: Path, ollama_model: str = "reader-lm:latest", fout: bool = False, skip_if_fout_exists: bool = True) -> str:
    assert fp.exists() and fp.is_file()

    fno: str = fp.name
    fno = fno[0:fno.rfind(".")] + "_readerlm.md"

    fw: Path = Path(fp.parent.resolve(), fno)

    if fw.exists() and skip_if_fout_exists:
        logger.debug(f"EXISTS -> SKIP: {fw.resolve()}")
        with open(fw, "r") as fin:
            return fin.read()

    prompt: str

    with open(fp, "r") as fin:
        prompt = fin.read().strip()

    # das muß in tokens (nicht chars) sein!
    # num_predict: int = len(prompt)  # wird ja nicht länger!

    resp: dict = ask_ollama_chat(  # type: ignore
        # repeat_penalty=1.9,
        system_prompt=None,
        streamed=True,
        prompt=prompt,
        model=ollama_model,
        evict=False,
        # temperature=0.1,
        # top_k=10,
        # top_p=0.1,
        num_predict=-2,
        seed=0,
        print_response=False,
        print_chunks_when_streamed=False,
        print_options=True,
        print_msgs=False
    )

    logger.debug(json.dumps(resp, indent=4, default=str, sort_keys=False))

    new_text: str = resp["message"]["content"]
    ret = new_text

    logger.debug(f"\n{ret}")

    if fout:
        with open(fw, "w") as fwout:
            fwout.write(ret)

        logger.debug(f"Written to: {fw.resolve()}")

    return ret
