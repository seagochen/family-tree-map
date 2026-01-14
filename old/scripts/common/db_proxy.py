from collections.abc import Iterator
from datetime import datetime
from contextlib import AbstractContextManager
from pathlib import PurePath
from typing import NamedTuple

import psycopg

from .items import Label, DatasetType, Sample


# Little chance that switch to a DB other than Postgres, so we don't add a
# class hierarchy here.
class DBProxy:
    class SpatialModelInfo(NamedTuple):
        model_id: int
        feature_size: int

    def connection(
        self,
        host: str,
        db: str,
        user: str,
        secret: str,
        port: None | int = None
    ) -> AbstractContextManager:
        self.conn = psycopg.connect(
            f"postgresql://{user}:{secret}@{host}/{db}", autocommit = True
        )
        return self.conn

    def transcation(self) -> AbstractContextManager:
        return self.conn.transaction()

    def spatial_model_info(
        self,
        architecture: str,
        variant: str,
        training_started: datetime,
    ) -> SpatialModelInfo:
        # SPROCの呼び出すSQLを用意
        #   引数リストの付いていなければ、追加しておく
        rec = self.conn.execute(
            "SELECT * FROM "
                "get_spatial_model_info(%s::spatial_model_type, %s, %s)",
            (architecture, variant, training_started),
        ).fetchone()
        if rec is None:
            raise LookupError(
                f"Spatial model not found: architecture({architecture}), "
                f"variant({variant}), timestamp({training_started})"
            )
        return __class__.SpatialModelInfo(*rec) # type: ignore[name-defined]

    # Return:
    #   True if the video is added, and False if it's already present in DB.
    def add_video(self, video_path: PurePath) -> bool:
        rec = self.conn.execute(
            "SELECT add_video(%s)", (str(video_path),)
        ).fetchone()
        assert rec is not None
        return rec[0]

    # Return:
    #   True if the clip is added, and False if it's already present in DB.
    def add_clip(
        self, video_path: PurePath, start_time: int, label: Label
    ) -> bool:
        rec = self.conn.execute(
            "SELECT add_clip(%s, %s, %s)", (str(video_path), start_time, label)
        ).fetchone()
        assert rec is not None
        return rec[0]

    # Parameters:
    #   Use strings to keep float numbers without precision losses.
    # Return:
    #   ID of the extraction parameters
    def register_extraction_parameters(
        self,
        spatial_model: int,
        detection_threshold: str,
        fn_area_threshold: str,
        tp_iou_threshold: str,
    ) -> int:
        rec = self.conn.execute(
            "SELECT register_extraction_parameters(%s, %s::numeric, %s::numeric"
                ", %s::numeric)",
            (
                spatial_model,
                detection_threshold,
                fn_area_threshold,
                tp_iou_threshold,
            ),
        ).fetchone()
        assert rec is not None
        return rec[0]

    # Parameters:
    #   feature_vector:
    #     Cannot be of Sequence type since Postgres API requires an array to be
    #       passed with list.
    # Return:
    #   True if the feature is inserted, and False if it's already present and
    #   is updated.
    def save_frame_feature(
        self,
        video_id: int,
        frame_time: int,
        extract_params: int,
        feature_vector: list[float],
        label: Label,
    ):
        self.conn.execute(
            "SELECT save_frame_feature(%s, %s, %s, %s::real[], %s)",
            (video_id, frame_time, extract_params, feature_vector, label),
        )

    # Return:
    #   Generator that pops up tuples of:
    #     (video ID, video file path, clip start timestamp)
    def pending_clips(
        self, extract_params: int
    ) -> Iterator[tuple[int, PurePath, int]]:
        yield from self.conn.execute(
            "SELECT * FROM get_pending_clips(%s)", (extract_params,)
        )

    # Return:
    #   Split ID
    def split_clips(
        self,
        extract_params: int,
        valid_feature_ratio: str,
        training_set_ratio: str,
        validation_set_ratio: str,
    ) -> int:
        rec = self.conn.execute("SELECT split_clips(%s, %s, %s, %s)", (
            extract_params,
            valid_feature_ratio,
            training_set_ratio,
            validation_set_ratio,
        )).fetchone()
        assert rec is not None
        return rec[0]

    # Return:
    #   Generator that pops up tuples of:
    #     (features of all frames in the clip, clip label)
    def dataset(
        self, split_id: int, category: DatasetType
    ) -> Iterator[Sample]:
        yield from self.conn.execute(
            "SELECT * FROM get_dataset(%s, %s)", (split_id, category),
        )

    # Return:
    #   Value of the configuration item.
    def get_config(self, item: str) -> str:
        rec = self.conn.execute("SELECT get_config(%s)", (item,)).fetchone()
        assert rec is not None
        return rec[0]

    def get_config_as_int(self, item: str) -> int:
        rec = self.conn.execute(
            "SELECT get_config_as_integer(%s)", (item,)
        ).fetchone()
        assert rec is not None
        return rec[0]

    def get_config_as_float(self, item: str) -> float:
        rec = self.conn.execute(
            "SELECT get_config_as_float(%s)", (item,)
        ).fetchone()
        assert rec is not None
        return rec[0]

    def get_config_as_time(self, item: str) -> datetime:
        rec = self.conn.execute(
            "SELECT get_config_as_time(%s)", (item,)
        ).fetchone()
        assert rec is not None
        return rec[0]
