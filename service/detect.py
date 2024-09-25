from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from service.inference import inference_service
from service.openface import openface_service

TEMP_PATH = Config.get_temp_path()


class DetectService:
    def __init__(self):
        self.openface_service = openface_service
        self.inference_service = inference_service

    async def update_batch_feature(self, feature_files, batch_file):
        df_list = []
        for csv_file in feature_files:
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

    async def image_detect(self, image_paths, batch_no):
        feature_files = await self.feature_extraction_by_images(image_paths)
        batch_dir = Path(f"{TEMP_PATH}/img/{batch_no}")
        batch_file = batch_dir / "batch_feature.csv"
        batch_feature = await self.update_batch_feature(feature_files, batch_file)
        # print(f"batch_feature: {batch_feature}")
        fkps, gaze = await self.inference_service.get_visual_data(batch_feature)
        visual_input = np.concatenate((fkps, gaze), axis=1)
        print(f"visual_input shape: {visual_input.shape}")
        visual_input = np.resize(
            visual_input,
            (2, visual_input.shape[0], visual_input.shape[1], visual_input.shape[2]),
        )
        detect_dict = await self.inference_service.visual_inference(visual_input)
        return detect_dict


detect_service = DetectService()
