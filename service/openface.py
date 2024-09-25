import os
import subprocess
from pathlib import Path

from config import Config
TEMP_PATH = Config.get_temp_path()

class OpenFaceService:
    def __init__(self):
        pass


    async def feature_extraction_by_video(self, video, out_dir=None):

        return

    async def feature_extraction_by_images(self, image_paths, out_dir=None):
        if os.name == "nt":
            command = r"D:\Programs\OpenFace_2.2.0_win_x64\FaceLandmarkImg.exe"
        else:
            command = "FaceLandmarkImg"
        image_args = []
        feature_files = []
        for image_path in image_paths:
            image_args.append("-f")
            image_args.append(image_path)
            image_name = Path(image_path).name
            image_stem = Path(image_path).stem

            if not out_dir:
                image_directory = Path(image_path).parent
                out_dir = image_directory / "processed"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_dir = out_dir.as_posix()

            feature_file = f"{out_dir}/{image_stem}.csv"
            feature_files.append(feature_file)
        feature_args = ["-3Dfp", "-pose", "-aus", "-gaze"]
        out_dir_args = ["-out_dir", out_dir]
        args = image_args + feature_args + out_dir_args

        print(f"command: {command} , args: {args}")

        try:
            result = subprocess.run([command] + args, check=True, capture_output=True, text=True)
            # 打印输出结果
            print(result.stdout)
            return feature_files
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            print(e.output)
            return []


openface_service = OpenFaceService()

