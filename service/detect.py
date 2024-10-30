import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.constants import ALL_LABELS_DESC_DICT
from config import Config, settings
from service.face import video_fp_feature, hdr, infer_video_model
from service.inference import inference_service
from service.openface import openface_service

TEMP_PATH = Config.get_temp_path()


class DetectService:
    def __init__(self):
        self.openface_service = openface_service
        self.inference_service = inference_service

    def padding(self, data, pad_size=120):
        if data.shape[0] < pad_size:
            size = tuple()
            size = size + (pad_size,) + data.shape[1:]
            padded_data = np.zeros(size)
            padded_data[: data.shape[0]] = data
        else:
            padded_data = data[:pad_size]
        return padded_data

    async def update_batch_feature(self, feature_files, batch_file):
        df_list = []
        for csv_file in feature_files:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                df_list.append(df)
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.to_csv(batch_file, mode="a", header=False, index=False)
        return pd.read_csv(batch_file, header=None)

    async def feature_extraction_by_images(self, image_paths):
        feature_files = await self.openface_service.feature_extraction_by_images(
            image_paths
        )
        return feature_files

    async def feature_extraction_by_video(self, video_path):
        feature_files = await self.openface_service.feature_extraction_by_video(
            [video_path]
        )
        return feature_files

    async def image_detect(self, image_paths, batch_no):
        feature_files = await self.feature_extraction_by_images(image_paths)
        batch_dir = Path(f"{TEMP_PATH}/img/{batch_no}")
        batch_file = batch_dir / "batch_feature.csv"
        batch_feature = await self.update_batch_feature(feature_files, batch_file)
        # print(f"batch_feature: {batch_feature}")
        fkps, gaze = await self.inference_service.get_visual_data(
            batch_feature, by_type="image"
        )
        visual_input = np.concatenate((fkps, gaze), axis=1)
        print(f"visual_input shape: {visual_input.shape}")
        # 为了适应视频训练图片的输入参数，填充至1800的时长 0填充
        # visual_input = await self.inference_service.visual_padding(visual_input)
        # 重复填充图片特征到1800的时长
        visual_input = np.resize(
            visual_input,
            (1800, visual_input.shape[1], visual_input.shape[2]),
        )

        # 为了支持模型的输入的batch，重复添加一组数据
        visual_input = np.resize(
            visual_input,
            (2, visual_input.shape[0], visual_input.shape[1], visual_input.shape[2]),
        )
        detect_dict = await self.inference_service.visual_inference(visual_input)
        return detect_dict

    async def multi_class_detect(self, image_paths, batch_no):
        feature_files = await self.feature_extraction_by_images(image_paths)
        batch_dir = Path(f"{TEMP_PATH}/img/{batch_no}")
        batch_file = batch_dir / "batch_feature.csv"
        batch_feature = await self.update_batch_feature(feature_files, batch_file)
        data = pd.read_csv(batch_file)
        data = self.padding(data.iloc[:, 2:30], pad_size=28)
        print(f"data shape: {data.shape}")
        features = np.array(data, np.float32)
        print(f"features: {features.shape}")
        features = torch.tensor(features).view(1, 28, 28).unsqueeze(0)
        detect_dict = await self.inference_service.multi_class_inference(features)
        return detect_dict

    async def video_detect(self, video_path, batch_no):
        feature_files = await self.feature_extraction_by_video(video_path)
        batch_dir = Path(f"{TEMP_PATH}/video/{batch_no}")
        batch_file = batch_dir / "batch_feature.csv"
        batch_feature = await self.update_batch_feature(feature_files, batch_file)
        # print(f"batch_feature: {batch_feature}")
        fkps, gaze = await self.inference_service.get_visual_data(
            batch_feature, by_type="video"
        )
        visual_input = np.concatenate((fkps, gaze), axis=1)
        print(f"visual_input shape: {visual_input.shape}")
        # 为了适应视频训练图片的输入参数，填充至1800的时长 0填充
        # visual_input = await self.inference_service.visual_padding(visual_input)
        # 重复填充图片特征到1800的时长
        visual_input = np.resize(
            visual_input,
            (1800, visual_input.shape[1], visual_input.shape[2]),
        )

        # 为了支持模型的输入的batch，重复添加一组数据
        visual_input = np.resize(
            visual_input,
            (2, visual_input.shape[0], visual_input.shape[1], visual_input.shape[2]),
        )
        detect_dict = await self.inference_service.visual_inference(visual_input)
        depressed_id = detect_dict["depressed_id"]
        detect_list = []
        # 是否用多分类模型推理
        if settings.MULTI_CLASS_METHOD == "one2one":
            data = pd.read_csv(batch_file)
            data = self.padding(data.iloc[:, 2:30], pad_size=28)
            print(f"data shape: {data.shape}")
            features = np.array(data, np.float32)
            print(f"features: {features.shape}")
            features = torch.tensor(features).view(1, 28, 28).unsqueeze(0)
            class_list = ["F20", "F31", "F32", "F41", "F42"]
            for f_class in class_list:
                f_class_detect_dict = (
                    await self.inference_service.multi_one2one_inference(
                        f_class, features
                    )
                )
                if f_class == "F32":
                    if depressed_id:
                        detect_list.append(f_class_detect_dict)
                    else:
                        state = "normal"
                        description = ALL_LABELS_DESC_DICT[state]
                        detect_list.append(
                            {
                                "index": 0,
                                "state": state,
                                "description": description,
                                "class_str": f_class,
                                "class_name": ALL_LABELS_DESC_DICT[f_class],
                            }
                        )
                else:
                    detect_list.append(f_class_detect_dict)

        detect_dict.update({"detect_list": detect_list})
        return detect_dict

    async def video_detect_new(self, video_path, batch_no):
        feature_files = await self.feature_extraction_by_video(video_path)
        batch_dir = Path(f"{TEMP_PATH}/video/{batch_no}")
        batch_file = batch_dir / "batch_feature.csv"
        batch_feature = await self.update_batch_feature(feature_files, batch_file)
        fp_filename = batch_dir / "fp_feature.csv"
        video_fp_feature(video_path, fp_filename)
        hdr_path = batch_dir / "video_capture.csv"
        hdr(fp_filename, hdr_path)
        min_video_score, video_scores = infer_video_model(hdr_path)

        print(f"视频频模型结束... video_scores: {video_scores}")
        print(f"视频频模型结束... min_video_score: {min_video_score}")
        # 转换为百分制
        centesimal_min_video_score = min_video_score / 24 * 100
        centesimal_video_scores = [int((x / 24) * 100) for x in video_scores]
        threshold = 35
        count_gt_threshold = sum(1 for x in centesimal_video_scores if x > threshold)

        if count_gt_threshold > 2:
            depressed_id = 1
            depressed_state = "抑郁"
            depressed_score = int(
                sum(x for x in centesimal_video_scores if x > threshold)
                / count_gt_threshold
            )
        elif count_gt_threshold == 2:
            depressed_id = 0
            depressed_state = "正常，有抑郁风险。"
            depressed_score = int(
                sum(centesimal_video_scores) / len(centesimal_video_scores)
            )
        else:
            depressed_id = 0
            depressed_state = "正常"
            depressed_score = int(
                sum(centesimal_video_scores) / len(centesimal_video_scores)
            )

        detect_dict = {
            "depressed_id": depressed_id,
            "depressed_state": depressed_state,
            "depressed_index": depressed_score,
            "depressed_score": depressed_score,
            "depressed_score_list": centesimal_video_scores
        }

        depressed_id = detect_dict["depressed_id"]
        detect_list = []
        # 是否用多分类模型推理
        if settings.MULTI_CLASS_METHOD == "one2one":
            data = pd.read_csv(batch_file)
            data = self.padding(data.iloc[:, 2:30], pad_size=28)
            print(f"data shape: {data.shape}")
            features = np.array(data, np.float32)
            print(f"features: {features.shape}")
            features = torch.tensor(features).view(1, 28, 28).unsqueeze(0)
            class_list = ["F20", "F31", "F32", "F41", "F42"]
            for f_class in class_list:
                f_class_detect_dict = (
                    await self.inference_service.multi_one2one_inference(
                        f_class, features
                    )
                )
                if f_class == "F32":
                    if depressed_id:
                        detect_list.append(f_class_detect_dict)
                    else:
                        state = "normal"
                        description = ALL_LABELS_DESC_DICT[state]
                        detect_list.append(
                            {
                                "index": 0,
                                "state": state,
                                "description": description,
                                "class_str": f_class,
                                "class_name": ALL_LABELS_DESC_DICT[f_class],
                            }
                        )
                else:
                    detect_list.append(f_class_detect_dict)

        detect_dict.update({"detect_list": detect_list})
        return detect_dict


detect_service = DetectService()
