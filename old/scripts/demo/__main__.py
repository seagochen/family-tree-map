from argparse import ArgumentParser
from pathlib import PurePath
import sys

from .app import App


def main() -> int:
    ap = ArgumentParser()
    ap.add_argument("--stream")
    ap.add_argument("--spatial-model", type = PurePath)
    ap.add_argument(
        "--spatial-detection-threshold", default = "0.1", type = PurePath
    )
    ap.add_argument("--temporal-model", type = PurePath)
    ap.add_argument("--db-host")
    ap.add_argument("--db-name")
    ap.add_argument("--db-user")
    ap.add_argument("--db-secret")

    return 1 - int(App().run(ap.parse_args()))


sys.exit(main())
