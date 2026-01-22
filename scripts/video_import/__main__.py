from argparse import ArgumentParser
from pathlib import PurePath
import sys

from .video_importer import VideoImporter


def main() -> int:
    ap = ArgumentParser()
    ap.add_argument("--data-root", type = PurePath)
    ap.add_argument("--clip-tsv", type = PurePath)
    ap.add_argument("--db-host")
    ap.add_argument("--db-name")
    ap.add_argument("--db-user")
    ap.add_argument("--db-secret")

    return 1 - int(VideoImporter().run(ap.parse_args()))


sys.exit(main())
