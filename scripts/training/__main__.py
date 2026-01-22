from argparse import ArgumentParser
from pathlib import PurePath
import sys

from .trainer import Trainer


def main() -> int:
    ap = ArgumentParser()
    ap.add_argument("--db-host")
    ap.add_argument("--db-name")
    ap.add_argument("--db-user")
    ap.add_argument("--db-secret")
    ap.add_argument("--output-directory", type = PurePath)

    return 1 - int(Trainer().run(ap.parse_args()))


sys.exit(main())
