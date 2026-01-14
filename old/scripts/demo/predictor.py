from collections.abc import Sequence
from pathlib import PurePath
from typing import NamedTuple

import cv2
import torch
from numpy import ndarray

from scripts.common.items import BBox
from scripts.feature_extraction.feature_extractor import FeatureExtractor
from scripts.training.temporal_model import TemporalModel


class Predictor:
    class Prediction(NamedTuple):
        frame_index: int
        bboxes: tuple[BBox, ...]
        score: float
        spatial_confidences: tuple[float]  # 空间模型的置信度列表
        types: tuple[int]

    def __init__(
            self,
            spatial_model_file: PurePath,
            spatial_detection_threshold: float,
            temporal_model_file: PurePath,
    ):
        self.feature_extractor = FeatureExtractor(
            spatial_model_file, spatial_detection_threshold
        )
        self.temporal_model: TemporalModel = torch.load(str(temporal_model_file))

    # 计算帧差
    @staticmethod
    def compute_frame_diff(prev: ndarray, curr: ndarray) -> float:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr_gray, prev_gray)
        return float(ndarray.mean(diff) / 255.0)

    # Parameters:
    #   clip_frames:
    #     Frames of a single clip. The frames must be sorted by their
    #     timestamps.
    def predict(self, clip_frames: Sequence[ndarray]) -> None | Prediction:
        with torch.inference_mode():
            # Note: If GPU memory is not enough, split the frames into multiple
            #   chunks.
            preds = self.feature_extractor.predict_and_extract(clip_frames)
            # 添加帧差
            features = []
            prev_frame = None
            for (frame, (_, vec, _, _)) in zip(clip_frames, preds):
                if prev_frame is None:
                    frame_diff = 0.0
                else:
                    frame_diff = self.compute_frame_diff(prev_frame, frame)
                prev_frame = frame
                vec.append(frame_diff)
                features.append(vec)

            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            labels = self.temporal_model(features_tensor)
            # Pick the output (log softmax) at the last time step
            if torch.isnan(labels).any():
                print("模型输出为 NaN，跳过当前 clip")
                return None

            non_fire_score, fire_score = labels[0]
            probs = torch.exp(labels[0])  # 转为 softmax 概率
            fire_prob = probs[1].item()
            non_fire_prob = probs[0].item()

            print(f"[火災検知预测] fire={fire_prob:.4f}, non_fire={non_fire_prob:.4f}")
            if (fire_score <= non_fire_score):
                return None

            # Fire detected
            # Choose the 1st frame with a detection or the frame in the middle
            # if there is no detection in any frame (since we're less confident,
            # just take an average).
            frame_idx = next((i for i, (b, _, _, _) in enumerate(preds) if b), None)

            # 返回所有的标注框，原先只返回第一个标注框
            all_bboxes = []
            all_scores = []
            all_types = []
            for detection, frame in zip(preds, clip_frames):
                # 获取这一帧的标注框
                bboxes, vec, scores, types = detection
                if bboxes:
                    all_bboxes.append(list(bboxes))
                    all_scores.append(list(scores))
                    all_types.append(list(types))
                else:
                    all_bboxes.append([])
                    all_scores.append([])
                    all_types.append([])

            # 如果没有任何检测，返回 None
            if not all_bboxes:
                frame_idx = len(preds) // 2
                all_bboxes.append((0.0, 0.0, 1.0, 1.0), )

            return self.Prediction(frame_idx, tuple(all_bboxes), fire_score, all_scores, all_types)
