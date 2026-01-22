import functools
import os
import traceback
from argparse import Namespace
from collections.abc import Sequence

import cv2 as cv
import numpy as np
import torch
from numpy import ndarray

from scripts.common.db_proxy import DBProxy
from scripts.common.items import BBox, Label
from scripts.common.video_data import VideoData
from scripts.feature_extraction.feature_extractor import FeatureExtractor


class FeatureMaker:
    @staticmethod
    def compute_frame_diff(prev_frame: ndarray, curr_frame: ndarray) -> float:
        prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
        diff = cv.absdiff(curr_gray, prev_gray)
        return float(np.mean(diff) / 255.0)  # 归一化到 0~1

    # Return: If the run succeeds or not.
    def run(self, args: Namespace) -> bool:
        result = True
        try:
            db_proxy = DBProxy()
            with db_proxy.connection(
                    args.db_host, args.db_name, args.db_user, args.db_secret
            ):
                # Get configuration
                det_thr = db_proxy.get_config("extractor_detection_threshold")
                model_arch = db_proxy.get_config("extractor_model_architecture")
                model_training_started = db_proxy.get_config_as_time("extractor_model_timestamp")
                model_variant = db_proxy.get_config("extractor_model_variant")
                fn_area_threshold = db_proxy.get_config("extractor_fn_area_threshold")
                tp_iou_threshold = db_proxy.get_config("extractor_tp_iou_threshold")

                model_id, _ = db_proxy.spatial_model_info(
                    model_arch, model_variant, model_training_started
                )
                extract_params = db_proxy.register_extraction_parameters(
                    model_id, det_thr, fn_area_threshold, tp_iou_threshold
                )
                video_data = VideoData(args.data_root)
                extractor = FeatureExtractor(
                    args.extractor_model, float(det_thr)
                )
                self._process_pending_clips(
                    video_data,
                    extractor,
                    extract_params,
                    float(fn_area_threshold),
                    float(tp_iou_threshold),
                    db_proxy,
                )
        except KeyboardInterrupt:
            print("Interrupted by user")
        except BaseException:
            result = False
            print(f"Error: {traceback.format_exc()}")
        return result

    def _process_pending_clips(
            self,
            video_data: VideoData,
            extractor: FeatureExtractor,
            extraction_parameters: int,
            fn_area_threshold: float,
            tp_iou_threshold: float,
            db_proxy: DBProxy,
    ):
        frame_interval_ms = db_proxy.get_config_as_int("frame_sampling_interval")
        frame_cnt = db_proxy.get_config_as_int("clip_frame_count")

        # No transcation needed at this level because clips don't have to be
        # processed in one pass.
        for vid_id, vid_path, start_tm in db_proxy.pending_clips(
                extraction_parameters
        ):
            print(
                f"====== Make features for clip: video({vid_path}), "
                f"start_time({start_tm}) ======"
            )
            # Since we check only the 1st frame of a clip to decide if it's
            # pending or not, we need transactions for making sure either all
            # the frames or none is processed.
            with db_proxy.transcation():
                prev_image = None
                for i in range(frame_cnt):
                    frame_tm = start_tm + i * frame_interval_ms
                    img_path = video_data.image_path(vid_path, frame_tm)
                    if not os.path.exists(img_path):
                        print(f"图像文件不存在，跳过: {img_path}")
                        continue
                    image = cv.imread(str(img_path))
                    if image is None:
                        print(f"无法读取图像，跳过: {img_path}")
                        continue

                    # Calculate feature vector
                    print(f"predict_and_extract start: {img_path}")

                    if prev_image is None:
                        frame_diff = 0.0
                    else:
                        frame_diff = self.compute_frame_diff(prev_image, image)
                    prev_image = image

                    bboxes, vec, _, _ = extractor.predict_and_extract((image,))[0]
                    vec.append(frame_diff)  # <-- 添加这一行，附加帧差到特征向量尾部
                    vec_tensor = torch.tensor(vec, dtype=torch.float32)
                    if torch.isnan(vec_tensor).any():
                        print(f"特征向量中包含 NaN，图像路径: {img_path}")
                        raise ValueError(f"Feature vector contains NaN: {img_path}")

                    anno_path = video_data.annotation_path(vid_path, frame_tm)
                    if not os.path.exists(anno_path):
                        print(f"注解文件不存在，跳过并标记为 NON_FIRE: {anno_path}")
                        lbl = Label.NON_FIRE
                    else:
                        try:
                            # Decide with annotation information if the feature of the
                            # frame represents fire or not
                            anno_objs = video_data.annotated_objects(vid_path, frame_tm)
                            h, w = image.shape[:2]
                            canvas_sz = (w, h)

                            # 使用模型推测出来的火灾框和xml标注的框计算交并比，根据交并比给图像分类为fire或者non-fire
                            lbl = self._label_frame_feature(
                                bboxes, anno_objs, fn_area_threshold, tp_iou_threshold, canvas_sz
                            )
                            print(f"_label_frame_feature end: {img_path}, {lbl}")
                        except Exception as e:
                            print(f"解析注解文件失败，跳过并标记为 NON_FIRE: {anno_path}, 错误: {e}")
                            lbl = Label.NON_FIRE

                    db_proxy.save_frame_feature(
                        vid_id, frame_tm, extraction_parameters, vec, lbl
                    )
                    print(
                        f"Feature saved: frame_time({frame_tm}), "
                        f"label({lbl})"
                    )

    '''
    def _label_frame_feature(
            prediction: Sequence[BBox],
            annotation: Sequence[BBox],
            fn_area_threshold: float,
            tp_iou_threshold: float,
    ) -> Label:
    '''

    @staticmethod
    def _label_frame_feature(
            prediction: Sequence[BBox],
            annotation: Sequence[BBox],
            fn_area_threshold: float,
            tp_iou_threshold: float,
            canvas_sz: tuple[int, int],
    ) -> Label:
        # Use the default input size of YOLOv8
        # canvas_sz = (640, 640)  # width x height

        def iou_geometric(box1: BBox, box2: BBox) -> float:
            x1_min = box1[0]
            y1_min = box1[1]
            x1_max = box1[0] + box1[2]
            y1_max = box1[1] + box1[3]

            x2_min = box2[0]
            y2_min = box2[1]
            x2_max = box2[0] + box2[2]
            y2_max = box2[1] + box2[3]

            xi_min = max(x1_min, x2_min)
            yi_min = max(y1_min, y2_min)
            xi_max = min(x1_max, x2_max)
            yi_max = min(y1_max, y2_max)

            inter_w = max(0, xi_max - xi_min)
            inter_h = max(0, yi_max - yi_min)
            inter_area = inter_w * inter_h

            area1 = box1[2] * box1[3]
            area2 = box2[2] * box2[3]
            union_area = area1 + area2 - inter_area
            return inter_area / union_area if union_area != 0 else 0.0

        # Note:
        #   Union of 0 annotated fires is treated as an empty polygon, whose
        #   area is 0.
        if prediction:
            # Fire detected by the spatial model
            #   The feature represents fire if predicted fire is close to
            #   annotated fire:
            #     Let P to be the union (polygon) of all predicted bboxes,
            #         T to be the union (polygon) of all annotated bboxes,
            #     "Close" is defined as: IoU(P, T) >= tp_iou_threshold
            #   Otherwise, the feature represents non-fire (frames with no
            #     annotated fire will fall into the branch naturally).
            iou = 0.0
            matched_by_center = False

            if annotation:
                # 计算所有预测框和标注框的 pair-wise IoU，并取最大值
                ious = [iou_geometric(pred, anno) for pred in prediction for anno in annotation]
                iou = max(ious)
                print(
                    f"_label_frame_feature iou_geometric: {iou}")

                if iou < tp_iou_threshold:
                    # Note: If the canvas size is too small, we may get 0 areas.
                    up = __class__._union_of_bboxes(prediction, canvas_sz)  # type: ignore[name-defined]
                    ua = __class__._union_of_bboxes(annotation, canvas_sz)  # type: ignore[name-defined]

                    # 交并比
                    iou = float(cv.countNonZero(cv.bitwise_and(up, ua))) \
                          / cv.countNonZero(cv.bitwise_or(up, ua))

                    pw, ph = canvas_sz
                    for pred in prediction:
                        cx_pred = (pred[0] + pred[2] / 2) * pw
                        cy_pred = (pred[1] + pred[3] / 2) * ph
                        for anno in annotation:
                            cx_anno = (anno[0] + anno[2] / 2) * pw
                            cy_anno = (anno[1] + anno[3] / 2) * ph
                            dist = ((cx_pred - cx_anno) ** 2 + (cy_pred - cy_anno) ** 2) ** 0.5
                            if dist < 15:
                                matched_by_center = True
                                break
                        if matched_by_center:
                            break
                    print(
                        f"_label_frame_feature iou: {iou}, tp_iou_threshold: {tp_iou_threshold}, center_match: {matched_by_center}")

                print(f"_label_frame_feature iou: {iou}, tp_iou_threshold: {tp_iou_threshold}")
                lbl = Label.FIRE if iou >= tp_iou_threshold else Label.NON_FIRE
        else:
            # No fire detected
            #   The feature represents fire if annotated fire is big:
            #     Let T is the union of all annotated bboxes,
            #         F is the area of the frame (= 1.0 since bbox coordinates
            #           are ratios).
            #     "Big" is defined as: Area(T) >= F * fn_area_threshold
            #   Otherwise, the feature represents non-fire (frames with no
            #     annotated fire will fall into the branch naturally).
            fire_area = cv.countNonZero(
                __class__._union_of_bboxes(annotation, canvas_sz)  # type: ignore[name-defined]
            ) if annotation else 0.0
            lbl = Label.FIRE if fire_area >= fn_area_threshold else Label.NON_FIRE
        return lbl

    # Description:
    #   Draw a list of boxes with non-black color on a black canvas.
    # Parameters:
    #   canvas_size: Width and height (unit: pixel).
    # Return:
    #   The drawn image.
    @staticmethod
    def _union_of_bboxes(
            bboxes: Sequence[BBox], canvas_size: tuple[int, int]
    ) -> ndarray:
        assert (bboxes)
        # bb_pix = ((
        #     round(b[0] * canvas_size[0]),
        #     round(b[1] * canvas_size[1]),
        #     round(b[2] * canvas_size[0]),
        #     round(b[3] * canvas_size[1]),
        # ) for b in bboxes)

        bb_pix = ((  # 修复 (x, y, w, h) -> (x1, y1, x2, y2)
            round(b[0] * canvas_size[0]),
            round(b[1] * canvas_size[1]),
            round((b[0] + b[2]) * canvas_size[0]),
            round((b[1] + b[3]) * canvas_size[1]),
        ) for b in bboxes)

        bg = np.zeros(canvas_size[::-1], dtype=np.uint8)

        return functools.reduce(
            functools.partial(cv.rectangle, color=1, thickness=cv.FILLED),
            bb_pix,
            bg,
        )
