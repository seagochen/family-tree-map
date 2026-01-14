from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import PurePath
from typing import NamedTuple

import cv2 as cv
from numpy import ndarray
from torch.nn import Module
from ultralytics import YOLO

from scripts.common.items import BBox


class Detection(NamedTuple):
    bbox: BBox
    score: float
    cls: int


class Prediction(NamedTuple):
    detections: list[Detection]


class SpatialModel(ABC):
    def __init__(self):
        self.input_size = (640, 640)

    def predict(
            self, images: Sequence[ndarray], threshold: float
    ) -> list[Prediction]:
        inputs = tuple(map(lambda i: self._preprocess(i), images))
        return self._predict_impl(inputs, threshold)

    @abstractmethod
    def module(self) -> Module:
        pass

    @abstractmethod
    def _predict_impl(
            self, images: Sequence[ndarray], threshold: float
    ) -> list[Prediction]:
        raise NotImplementedError

    def _preprocess(self, image: ndarray) -> ndarray:
        return image  # self._resize_image(image, self.input_size, (114, 114, 114))

    @staticmethod
    def _resize_image(
            image: ndarray,
            new_size: tuple[int, int],
            padding_color: tuple[int, int, int],
    ) -> ndarray:
        out_w, out_h = new_size[0], new_size[1]
        img_w, img_h = image.shape[1::-1]  # HxWxC
        scale = min(out_w / img_w, out_h / img_h)
        w = int(round(img_w * scale))
        h = int(round(img_h * scale))
        assert w <= out_w and h <= out_h
        # If the difference is an odd number, distribute the extra one to
        # right/bottom.
        left = (out_w - w) >> 1
        right = (out_w - w + 1) >> 1
        top = (out_h - h) >> 1
        bottom = (out_h - h + 1) >> 1
        img = cv.resize(image, (w, h), interpolation=cv.INTER_LINEAR)
        # Padding
        return cv.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv.BORDER_CONSTANT,
            value=padding_color,
        )


class SpatialModelYOLOv8(SpatialModel):
    def __init__(self, model_file: PurePath):
        super().__init__()
        self.yolo_model = YOLO(model_file)

    def module(self) -> Module:
        return self.yolo_model.model

    def _predict_impl(
            self, images: Sequence[ndarray], threshold: float
    ) -> list[Prediction]:
        print(f"执行 _predict_impl: 图像数={len(images)}")
        # 使用 predict 方法，避免重复前向传播
        results = self.yolo_model.predict(images, imgsz=640, agnostic_nms=True, conf=threshold, verbose=True,
                                          stream=False)
        print(f"预测完成: 结果数={len(results)}")

        preds: list[Prediction] = []
        for r in results:
            # ltrb => ltwh
            r.boxes.xyxyn[:, 2] -= r.boxes.xyxyn[:, 0]
            r.boxes.xyxyn[:, 3] -= r.boxes.xyxyn[:, 1]
            preds.append(Prediction([
                Detection(tuple(b.tolist()), c.item(), int(s.item())) \
                for b, c, s in zip(r.boxes.xyxyn, r.boxes.conf, r.boxes.cls)
            ]))
        return preds
