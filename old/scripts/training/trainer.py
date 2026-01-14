import copy
import itertools
import traceback
from argparse import Namespace
from pathlib import PurePath
from typing import Iterator

import torch
from torch import Tensor
from torch.nn import NLLLoss
from torch.optim import SGD

from scripts.common.db_proxy import DBProxy
from scripts.common.items import DatasetType, Label, Sample
from scripts.training.temporal_model import TemporalModel


class Trainer:
    # Return: If the run succeeds or not.
    def run(self, args: Namespace) -> bool:
        result = True
        try:
            db_proxy = DBProxy()
            with db_proxy.connection(
                    args.db_host, args.db_name, args.db_user, args.db_secret
            ):
                self.train(db_proxy, args.output_directory)
        except KeyboardInterrupt:
            print("Interrupted by user")
        except BaseException:
            result = False
            print(f"Error: {traceback.format_exc()}")
        return result

    def train(self, db_proxy: DBProxy, output_directory: PurePath):
        # Get configuration
        det_thr = db_proxy.get_config("extractor_detection_threshold")
        em_arch = db_proxy.get_config("extractor_model_architecture")
        em_variant = db_proxy.get_config("extractor_model_variant")
        em_training_started = db_proxy.get_config_as_time("extractor_model_timestamp")
        fn_area_threshold = db_proxy.get_config("extractor_fn_area_threshold")
        tp_iou_threshold = db_proxy.get_config("extractor_tp_iou_threshold")
        valid_feat_ratio = db_proxy.get_config("temporal_valid_feature_ratio")
        training_ratio = db_proxy.get_config("dataset_training_ratio")
        validation_ratio = db_proxy.get_config("dataset_validation_ratio")
        batch_size = db_proxy.get_config_as_int("training_batch_size")
        lr = db_proxy.get_config_as_float("training_learning_rate")
        weight_decay = db_proxy.get_config_as_float("training_weight_decay")
        dropout = db_proxy.get_config_as_float("training_dropout")
        num_epoches = db_proxy.get_config_as_int("training_number_of_epoches")

        em_info = db_proxy.spatial_model_info(
            em_arch, em_variant, em_training_started
        )
        extract_params = db_proxy.register_extraction_parameters(
            em_info.model_id, det_thr, fn_area_threshold, tp_iou_threshold
        )
        # 将有效的clips放到datasets表中
        split_id = db_proxy.split_clips(
            extract_params,
            valid_feat_ratio,
            training_ratio,
            validation_ratio,
        )

        # Training loop
        # (https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
        model = TemporalModel(em_info.feature_size + 1, 512, 2, dropout)
        loss_func = NLLLoss()
        optimizer = SGD(model.parameters(), lr, weight_decay=weight_decay)
        max_acc = -1.0
        best_weights = None
        for i in range(num_epoches):
            # 取出有效的clips
            # 从 clips 中找出起点 (video_id, start_time)
            # 在对应 clip 时间段 [start_time, start_time + duration) 内
            # 是否至少有 7 帧满足条件的特征features存在
            # 把 features.vector 中每 10 帧组成一条 Clip（按时间和 label 匹配），用于训练 temporal model
            train_data = db_proxy.dataset(split_id, DatasetType.TRAINING)
            # Train
            model.train()
            while batched_data := self._fetch_batch(train_data, batch_size):
                inputs, truths = batched_data
                model.zero_grad()
                outputs = model(inputs)
                loss = loss_func(outputs, truths)
                loss.backward()
                assert optimizer
                optimizer.step()

            # Validate
            valid_data = db_proxy.dataset(split_id, DatasetType.VALIDATION)
            correct = 0
            total = 0
            assert model
            model.eval()
            with torch.inference_mode():
                while batched_data := self._fetch_batch(
                        valid_data, batch_size, False
                ):
                    inputs, truths = batched_data
                    total += inputs.size(0)
                    outputs = model(inputs)
                    # Pick only the output at the last time step
                    preds = outputs.argmax(1)
                    correct += (preds == truths).sum().item()
            acc = correct / total
            if acc > max_acc:
                max_acc = acc
                # Keep only the state dict
                best_weights = copy.deepcopy(model.state_dict())

        # Save model (network structure and weights) to a .pt file
        if best_weights:
            assert model
            model.load_state_dict(best_weights)
            torch.save(model, f"{output_directory}/temporal_model.pt")

        fire_cnt, non_fire_cnt = 0, 0
        for f, label in db_proxy.dataset(split_id, DatasetType.TRAINING):
            if label == Label.FIRE:
                fire_cnt += 1
            else:
                non_fire_cnt += 1
        print(f"[训练集分布] fire={fire_cnt}, non_fire={non_fire_cnt}")

    # Parameters:
    #   drop_last: Drop the last mini-batch if its size is smaller than batch
    #     size.
    # Return:
    #   (Inputs, Truths)
    #     Inputs: Clip features
    #       Shape = (batch, seq, feature_size)
    #     Truths: Ground truths of "Inputs"
    #       Shape = (batch, )
    #       label: 0(non_fire), 1(fire)
    @staticmethod
    def _fetch_batch(
            dataset: Iterator[Sample], batch_size: int, drop_last=True
    ) -> None | tuple[Tensor, Tensor]:
        target_seq_len = 10  # 固定目标序列长度

        samples = tuple(itertools.islice(dataset, batch_size))
        num_samples = len(samples)
        if num_samples == 0 or (num_samples < batch_size and drop_last):
            return None

        features, labels = zip(*samples)
        padded_features = []
        for i, f in enumerate(features):
            f_tensor = torch.tensor(f, dtype=torch.float32)
            if torch.isnan(f_tensor).any():
                print(f"[警告] 第 {i} 个样本中包含 NaN 特征")

            seq_len, feat_dim = f_tensor.shape
            if seq_len > target_seq_len:
                f_tensor = f_tensor[:target_seq_len]  # 截断
            elif seq_len < target_seq_len:
                pad_len = target_seq_len - seq_len
                pad_tensor = torch.zeros((pad_len, feat_dim), dtype=torch.float32)
                f_tensor = torch.cat((f_tensor, pad_tensor), dim=0)  # 填充

            padded_features.append(f_tensor)

        features_tensor = torch.stack(padded_features)  # Shape: (batch_size, target_seq_len, feature_dim)
        labels_tensor = torch.tensor([Label.to_int(l) for l in labels], dtype=torch.long)

        return features_tensor, labels_tensor
