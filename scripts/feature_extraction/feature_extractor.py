from collections.abc import Sequence
from pathlib import PurePath

import torch
from numpy import ndarray
from torch import Tensor
from torch.nn import Module

from scripts.common.items import BBox
from scripts.common.model import Detection, SpatialModelYOLOv8


# (a) vs (b):
#     (a): Pooling spatial feature map of all fires in an image (Kim's paper)
#     (b): Tracking and classifying each fire's feature maps independently
#   Go with pooling. Fire has no clear-cut edges and shapes, so it's hard to
#   separate it into independent parts (flames and/or smokes) in many cases,
#   i.e., we can be sure that whether or not there is a fire, but cannot give
#   an exact number of how many pieces of flames/smokes there are. Therefore,
#   it's more of a classification task than a detection one.
class FeatureExtractor:
    def __init__(self, model_file: PurePath, detection_threshold: float):
        self.model = SpatialModelYOLOv8(model_file)
        self.detect_threshold = detection_threshold
        self.feature_maps: list[Tensor] = []
        self.hook_handles = self._inspect_hidden_outputs(self.model.module(), (15, 18, 21), self.feature_maps)

    # Parameters:
    #   images: A batch of input images
    # Return:
    #   Detected bounding boxes and the extracted feature vector of each image
    #   in the batch
    def predict_and_extract(
            self, images: Sequence[ndarray]
    ) -> list[tuple[tuple[BBox, ...], list[float]]]:
        # Inference and get feature maps with the spatial model
        self.feature_maps.clear()
        # 移除旧 Hook 并重新注册
        if hasattr(self, 'hook_handles'):
            for handle in self.hook_handles:
                handle.remove()
        self.hook_handles = self._inspect_hidden_outputs(self.model.module(), (15, 18, 21), self.feature_maps)

        predictions = self.model.predict(images, self.detect_threshold)
        print("[调试] predictions =", predictions)
        if len(self.feature_maps) > 3:
            print(f"警告: 捕获过多特征图 ({len(self.feature_maps)})，只保留最后 3 个")
            self.feature_maps = self.feature_maps[-3:]
        if len(self.feature_maps) != 3:
            raise ValueError(f"预期捕获 3 个特征图，但实际为 {len(self.feature_maps)}")
        features = tuple(self.feature_maps)
        self.feature_maps.clear()

        ret = []
        assert len(predictions) == features[0].shape[0], f"预测数 {len(predictions)} != 特征批次 {features[0].shape[0]}"
        # For each frame in the batch
        for p, f in zip(
                predictions,
                zip(*features),  # Transpose to pop all features in a batch each time
        ):
            if not p.detections:
                # No fire detected
                #   Treat this frame as a negative detection with:
                #     A bbox covering the enitre image
                #     A score = 1 - fire_detection_threshold / 2
                #   Rational:
                #     There might be some fire detected if we lower the
                #     detection threshold, so we can think of the case as a
                #     single fire being detected with a score lower than the
                #     threshold, which is in range (0, fire_detection_threshold)
                #     (assuming multiple fires are possible would make things
                #     complicated, so we just keep it simple to assume only one
                #     fire). We can use the middle point of the score range as
                #     the average detection score of the imagined fire, so the
                #     score of the frame being background is:
                #       1 - avg_score_of_imagined_fire
                #         => 1 - fire_detection_threshold / 2
                p.detections.append(Detection(
                    (0.0, 0.0, 1.0, 1.0), 1.0 - 0.5 * self.detect_threshold, 2
                ))

            print(f"_calculate_feature_vector start p.detections: {p.detections}")
            feature_vec = __class__._calculate_feature_vector(  # type: ignore[name-defined]
                p.detections, f
            )

            ret.append((tuple(d.bbox for d in p.detections), feature_vec, tuple(d.score for d in p.detections), tuple(d.cls for d in p.detections)))
        return ret

    @staticmethod
    def _calculate_feature_vector(
            detections: Sequence[Detection], layer_features: Sequence[Tensor]
    ) -> list[float]:
        '''
            The following pooling approaches is not adopted because it's not a
            global average, but an average of multiple local averages (i.e.,
            averaging within each box and then averaging across boxes):
              For each feature channel, compute ∑b (Pb * Fb), where:
                b is each detected fire, Pb is the fire's detection confidence;
                Fb = ∑p (Fp) / N, where p is each point in the bbox, Fp is the
                  feature value of the point;
                N is the number of points the bbox contains.
            The approach described in the paper (
              https://www.mdpi.com/2076-3417/9/14/2862
            ) is more of a total average, which treats all detected regions in
            the image as a whole fire. A more intuitive equation to express
            their idea is:
              1) For each feature channel, compute:
                (W1*P1 + W2*P2 + ... + Wn*Pn) / (W1 + W2 + ... + Wn),
                where:
                  P is the feature value at each point in a detected region;
                  W is the score of the bbox that contains that point;
                  The sum is done over each point in all the detected regions.
                Thinking it this way also gives us a natural answer to the
                question of how to handle overlapped bboxes: we can simply treat
                points in an overlapped part as different points, i.e., make
                them contribute to the average multiple times - each time with
                the same feature value and a different weight (score).
              2) Pooling within each feature channel at each feature level and
                then concatenate all the results into a single 1D vector.
        '''
        feat_vec = []
        assert (detections)

        for f in layer_features:
            # f has a shape of (C, H, W)
            # feat_sum = torch.zeros(f.size(0))
            # pt_cnt = torch.zeros(f.size(0))

            device = layer_features[0].device
            feat_sum = torch.zeros(f.size(0), device=device)
            pt_cnt = torch.zeros(f.size(0), device=device)

            for d in detections:
                l = max(0, min(f.size(2) - 1, round(d.bbox[0] * f.size(2))))
                r = max(0, min(f.size(2), round((d.bbox[0] + d.bbox[2]) * f.size(2))))
                t = max(0, min(f.size(1) - 1, round(d.bbox[1] * f.size(1))))
                b = max(0, min(f.size(1), round((d.bbox[1] + d.bbox[3]) * f.size(1))))
                # 跳过无效区域
                if r <= l or b <= t:
                    print(f"[Warning] Skipping detection due to invalid region: l={l}, r={r}, t={t}, b={b}")
                    continue

                det_rgn = f[:, t:b, l:r]

                # 确保 d.score 也在同一设备
                score = torch.tensor(d.score, device=device)
                feat_sum += (score * det_rgn).sum((1, 2))
                pt_cnt += d.score * (b - t) * (r - l)

            if pt_cnt.sum() == 0:
                print(f"[Warning] No valid regions found for this layer, using zero vector.")
                feature_chunk = torch.zeros_like(feat_sum)
            else:
                feature_chunk = feat_sum / pt_cnt

            # 检查是否含 NaN
            if torch.isnan(feature_chunk).any():
                print("🚨 警告：特征向量中仍包含 NaN！")
                print(f"  🔸 feat_sum: {feat_sum}")
                print(f"  🔸 pt_cnt:   {pt_cnt}")
                raise ValueError("Feature vector contains NaN after pooling")

            feat_vec.extend(feature_chunk.tolist())

        return feat_vec

    @staticmethod
    def _inspect_hidden_outputs(module: Module, layers: Sequence[int], outputs: list[Tensor]):
        handles = []

        def hook_fn(m, i, o):
            outputs.append(o)
            # print(f"Hook 触发: 层={i}, 输出 shape={o.shape}, 当前 feature_maps 长度={len(outputs)}")

        for i in layers:
            handle = module.model[i].register_forward_hook(hook_fn)
            handles.append(handle)
            # print(f"注册 Hook: 层={i}, Handle={id(handle)}")
        return handles  # 返回 handles 以便管理
