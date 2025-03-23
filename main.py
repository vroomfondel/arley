import os
import sys

from arley.config import log_settings, is_in_cluster
import arley.emailinterface.imapadapter
import arley.emailinterface.ollamaemailreply

# from pprint import pprint

from loguru import logger

def imaploop() -> int:
    res: Exception | None = arley.emailinterface.imapadapter.main(
        max_idle_unsuccess_in_sequence=None,
        max_idle_loops=None,
        timeout_per_idle_loop=10
    )

    if res:
        return 123

    return 0

def ollamaloop() -> int:
    res: Exception | None = arley.emailinterface.ollamaemailreply.main(
        timeout_per_loop=5,
        max_loop=None
    )

    if res:
        return 123

    return 0

def main():
    if is_in_cluster():
        logger.info("presumably running in cluster")
        logger.info(f"BUILDTIME: {os.getenv("BUILDTIME")}")
    else:
        logger.info("not running in cluster")

    log_settings()

    main_ret: int = 0

    if len(sys.argv) > 1:
        if sys.argv[1] == "IMAPLOOP":
            main_ret = imaploop()
        elif sys.argv[1] == "OLLAMALOOP":
            main_ret = ollamaloop()

    exit(main_ret)

if __name__ == "__main__":
    main()
