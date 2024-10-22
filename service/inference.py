import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.constants import DEPRESSED_STATE_DICT, ALL_LABELS_DICT, ALL_LABELS_DESC_DICT
from config import Config, settings

from models.convlstm import ConvLSTMVisual
from models.evaluator import Evaluator
from models.multi_classification import MultiClassNet
from utils import init_seed

TEMP_PATH = Config.get_temp_path()


class InferenceService:
    def __init__(self, weights_path, multi_class_weights_path):
        init_seed()
        self.visual_net = ConvLSTMVisual(
            input_dim=3,
            output_dim=256,
            conv_hidden=256,
            lstm_hidden=256,
            num_layers=4,
            activation="relu",
            norm="bn",
            dropout=0.5,
        )

        self.evaluator = Evaluator(
            feature_dim=256, output_dim=2, predict_type="phq-binary", num_subscores=8
        )
        print(f"load visual_net: {weights_path} ...")
        self.weights_path = weights_path
        self.checkpoint = torch.load(weights_path)
        self.visual_net.load_state_dict(self.checkpoint["visual_net"], strict=False)
        self.evaluator.load_state_dict(self.checkpoint["evaluator"], strict=False)
        self.visual_net.eval()
        self.evaluator.eval()
        torch.set_grad_enabled(False)
        print(f"load visual_net: {weights_path} done")

        print(f"load multi_class_weights: {multi_class_weights_path} ...")

        self.multi_class_net = MultiClassNet()
        self.multi_class_net.load_state_dict(torch.load(multi_class_weights_path))
        self.multi_class_net.eval()
        print(f"load multi_class_weights: {multi_class_weights_path} done")

    def pre_check(self, data_df):
        data_df = data_df.apply(pd.to_numeric, errors="coerce")
        data_np = data_df.to_numpy()
        data_min = data_np[np.where(~(np.isnan(data_np[:, 2:])))].min()
        data_df.where(~(np.isnan(data_df)), data_min, inplace=True)
        return data_df

    def load_all_feature(self, feature):
        all_feature_df = self.pre_check(feature)
        return all_feature_df

    def min_max_scaler(self, data):
        """recale the data, which is a 2D matrix, to 0-1"""
        return (data - data.min()) / (data.max() - data.min())

    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

    def get_feature_index(self, by_type):

        if by_type == "image":
            gaze_indexes = (2, 8)
            key_point_x_index = (298, 366)
            key_point_y_index = (366, 434)
            key_point_z_index = (434, 502)
            key_point_indexes = (key_point_x_index, key_point_y_index, key_point_z_index)
        else:
            gaze_indexes = (5, 11)
            key_point_x_index = (301, 369)
            key_point_y_index = (369, 437)
            key_point_z_index = (437, 505)
            key_point_indexes = (key_point_x_index, key_point_y_index, key_point_z_index)
        return gaze_indexes, key_point_indexes

    def load_gaze(self, all_feature, by_type):
        gaze_indexes, _ = self.get_feature_index(by_type)
        gaze_coor = (
            all_feature.iloc[:, gaze_indexes[0]:gaze_indexes[1]].to_numpy().reshape(len(all_feature), 2, 3)
        )
        padding = np.zeros((len(all_feature), 2, 3))
        gaze_coor = np.concatenate((gaze_coor, padding), axis=1)
        return gaze_coor

    def load_keypoints(self, all_feature, by_type):
        _, key_point_indexes = self.get_feature_index(by_type)
        # process into format TxVxC
        x_coor = self.min_max_scaler(
            all_feature[all_feature.columns[key_point_indexes[0][0]:key_point_indexes[0][1]]].to_numpy()
        )
        y_coor = self.min_max_scaler(
            all_feature[all_feature.columns[key_point_indexes[1][0]:key_point_indexes[1][1]]].to_numpy()
        )
        z_coor = self.min_max_scaler(
            all_feature[all_feature.columns[key_point_indexes[2][0]:key_point_indexes[2][1]]].to_numpy()
        )
        fkps_coor = np.stack([x_coor, y_coor, z_coor], axis=-1)
        return fkps_coor

    async def get_visual_data(self, batch_feature, by_type="image"):
        all_feature = self.load_all_feature(batch_feature)
        gaze = self.load_gaze(all_feature, by_type)
        fkps = self.load_keypoints(all_feature, by_type)
        return fkps, gaze

    async def visual_padding(self, data, pad_size=1800):
        if data.shape[0] != pad_size:
            size = tuple()
            size = size + (pad_size,) + data.shape[1:]
            padded_data = np.zeros(size)
            padded_data[: data.shape[0]] = data
        else:
            padded_data = data

        return padded_data

    async def visual_inference(self, visual_input):
        visual_input = torch.from_numpy(visual_input).type(torch.FloatTensor)
        print(f"visual_inference visual_input shape: {visual_input.shape}")
        visual_input = visual_input.permute(0, 3, 2, 1).contiguous()
        print(f"visual_inference visual_input permute shape: {visual_input.shape}")
        with torch.no_grad():
            visual_features = self.visual_net(visual_input)
            predictions = self.evaluator(visual_features)
        print(f"visual_inference predictions: {predictions}")
        depressed_index = predictions[0][1].item()
        score_pred = predictions.argmax(dim=-1)
        binary_pred = score_pred[0].item()
        # binary_pred 0: 正常，1：抑郁
        print(f"binary_pred: {binary_pred}")
        return {
            "depressed_id": binary_pred,
            "depressed_state": DEPRESSED_STATE_DICT[binary_pred],
            "depressed_index":  f'{depressed_index:.2%}',
        }

    async def multi_class_inference(self, input_data):
        inverted_labels_dict = {value: key for key, value in ALL_LABELS_DICT.items()}
        with torch.no_grad():
            predictions = self.multi_class_net(input_data)
            _, predicted = torch.max(predictions, 1)
            print(f"infer: output: {predictions}, {_, predicted}")

            label_id = int(predicted)
            state = inverted_labels_dict[label_id]
            description = ALL_LABELS_DESC_DICT[state]

            return {
                "index": label_id,
                "state": state,
                "description": description,
            }


inference_service = InferenceService(
    weights_path=settings.MODEL_WEIGHTS_PATH,
    multi_class_weights_path=settings.MODEL_MULTI_CLASS_WEIGHTS_PATH,
)
