from collections.abc import Iterator, Sequence
import os
from os import path
from pathlib import Path, PurePath
import xml.etree.ElementTree as ET

from .items import BBox, Label


class VideoData:
    def __init__(self, root: PurePath):
        self.root_path = root

    # Return:
    #   Generator that pops up paths of video folders.
    def videos(self) -> Iterator[PurePath]:
        for child in os.scandir(self.root_path):
            if not child.is_dir() or not self._contains_video(
                PurePath(child.path)
            ):
                continue
            yield PurePath(child.name)

    # Parameters:
    #   tsv:
    #     Why we accept a path here instead of just returning all clips:
    #       Support splitting clips into multiple chunks.
    #     Do not include a header row in the file.
    # Return:
    #   Generator that pops up tuples of:
    #     (video file path, clip start timestamp, clip label)
    def clips(self, tsv: PurePath) -> Iterator[tuple[PurePath, int, Label]]:
        with open(tsv) as f:
            for ln in f:
                vid_dir, start_tm, lbl = ln.removesuffix('\n').split('\t')
                yield PurePath(vid_dir), int(start_tm), Label(lbl)

    # Description: Validate that a frame exists
    def frame_exists(self, video_folder: PurePath, time: int) -> bool:
        return path.isfile(
            self.root_path / video_folder / "frames" / f"{time}.jpg",
        )

    # Return:
    #   Bounding boxes of annotated objects in the image
    def annotated_objects(
        self, video_folder: PurePath, frame_time: int
    ) -> list[BBox]:
        anno_path = self.annotation_path(video_folder, frame_time)
        root = ET.parse(anno_path).getroot()

        size = root.find('size')
        if size is None:
            raise LookupError(f"Element 'size' not found: file({anno_path})")
        img_w_txt = size.findtext('width')
        img_h_txt = size.findtext('height')
        if (None in (img_w_txt, img_h_txt)): # type: ignore[union-attr]
            raise LookupError(
                f"Width and/or height info not found: file({anno_path})"
            )
        img_w = int(img_w_txt) # type: ignore[arg-type, union-attr]
        img_h = int(img_h_txt) # type: ignore[arg-type, union-attr]

        boxes = []
        for bb in root.findall("./object/bndbox"):
            assert bb is not None
            c: list[int] = []
            for e in ["xmin", "ymin", "xmax", "ymax"]:
                txt = bb.findtext(e)
                if txt is None:
                    raise LookupError(
                        f"Coordinate {e} not found: file({anno_path})"
                    )
                c.append(int(txt)) # type: ignore[arg-type]
            bb_w = c[2] - c[0]
            bb_h = c[3] - c[1]
            if (
                c[0] < 0 or c[1] < 0 or c[2] > img_w or c[3] > img_h
                or bb_w <= 0 or bb_h <= 0
            ):
                raise ValueError(
                    f"Invalid bounding box: coordinates({c}), "
                    f"image_width({img_w}), image_height({img_h}), "
                    f"file({anno_path})"
                )
            boxes.append((
                c[0] / img_w, c[1] / img_h, bb_w / img_w, bb_h / img_h,
            ))
        return boxes

    # Return:
    #   The absolute path of the image file.
    def image_path(self, video_folder: PurePath, frame_time: int) -> PurePath:
        return self._frame_path(video_folder, frame_time).with_suffix(".jpg")

    # Return:
    #   The absolute path of the annotation file.
    def annotation_path(
        self, video_folder: PurePath, frame_time: int
    ) -> PurePath:
        return self._frame_path(video_folder, frame_time).with_suffix(".xml")

    # Return:
    #   The absolute path of the files of the frame without extension.
    def _frame_path(self, video_folder: PurePath, frame_time: int) -> PurePath:
        return self.root_path / video_folder / f"frames/{frame_time}"

    # Parameters:
    #   video_folder: Full path.
    @staticmethod
    def _contains_video(video_folder: PurePath) -> bool:
        # Check if the folder contains a regular file named video.*
        match = (e for e in Path(video_folder).glob("video.*") if e.is_file())
        return next(match, None) is not None
