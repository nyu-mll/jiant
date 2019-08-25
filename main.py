import logging as log

log.basicConfig(
    format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO
)  # noqa

import sys

from jiant.__main__ import main

# Global notification handler, can be accessed outside main() during exception handling.
EMAIL_NOTIFIER = None

if __name__ == "__main__":
    try:
        main(sys.argv[1:])
        if EMAIL_NOTIFIER is not None:
            EMAIL_NOTIFIER(body="Run completed successfully!", prefix="")
    except BaseException as e:
        # Make sure we log the trace for any crashes before exiting.
        log.exception("Fatal error in main():")
        if EMAIL_NOTIFIER is not None:
            import traceback

            tb_lines = traceback.format_exception(*sys.exc_info())
            EMAIL_NOTIFIER(body="".join(tb_lines), prefix="FAILED")
        raise e  # re-raise exception, in case debugger is attached.
        sys.exit(1)
    sys.exit(0)
