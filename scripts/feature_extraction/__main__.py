from argparse import ArgumentParser
from pathlib import PurePath
import sys

from .feature_maker import FeatureMaker


def main() -> int:
    ap = ArgumentParser()
    ap.add_argument("--data-root", type = PurePath)
    ap.add_argument("--extractor-model", type = PurePath)
    ap.add_argument("--db-host")
    ap.add_argument("--db-name")
    ap.add_argument("--db-user")
    ap.add_argument("--db-secret")

    return 1 - int(FeatureMaker().run(ap.parse_args()))


sys.exit(main())
