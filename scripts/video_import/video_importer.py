from argparse import Namespace
from pathlib import PurePath
import sys
import traceback

from ..common.db_proxy import DBProxy
from ..common.video_data import VideoData


class VideoImporter:
    # Return: If the run succeeds or not.
    def run(self, args: Namespace) -> bool:
        result = True
        try:
            db_proxy = DBProxy()
            with db_proxy.connection(
                args.db_host, args.db_name, args.db_user, args.db_secret
            ):
                video_data = VideoData(args.data_root)
                self.import_data(video_data, args.clip_tsv, db_proxy)
        except KeyboardInterrupt:
            print("Interrupted by user")
        except BaseException:
            result = False
            print(f"Error: {traceback.format_exc()}")
        return result

    # Import videos and clips
    def import_data(self, video_data: VideoData, clip_tsv: PurePath, db_proxy: DBProxy):
        # Import videos
        for path in video_data.videos():
            added = db_proxy.add_video(path)
            print(
                f"Video: path({path}), "
                f"result({'added' if added else 'skipped'})"
            )

        # Import clips
        frame_interval_ms = db_proxy.get_config_as_int("frame_sampling_interval")
        frame_cnt = db_proxy.get_config_as_int("clip_frame_count")
        for vid_path, start_tm, lbl in video_data.clips(clip_tsv):
            missing_frame = self._clip_frame_missing(
                vid_path, start_tm, frame_interval_ms, frame_cnt, video_data,
            )
            if missing_frame is not None:
                print(
                    f"Failed to import clip due to frame missing: "
                    f"video({vid_path}), start_time({start_tm})"
                    f", missing_frame({missing_frame})"
                )
                continue
            added = db_proxy.add_clip(vid_path, start_tm, lbl)
            print(
                f"Clip: video({vid_path}), start_time({start_tm}), label({lbl})"
                f", result({'added' if added else 'skipped'})"
            )

    @staticmethod
    def _clip_frame_missing(
        video_path: PurePath,
        start_time: int,
        frame_interval_ms: int,
        frame_count: int,
        video_data: VideoData,
    ) -> None | int:
        for i in range(frame_count):
            frame_tm = start_time + i * frame_interval_ms
            if not video_data.frame_exists(video_path, frame_tm):
                return frame_tm
        return None
