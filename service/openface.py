import os
import subprocess
import sys
from pathlib import Path

from config import Config
TEMP_PATH = Config.get_temp_path()

class OpenFaceService:
    def __init__(self):
        pass

    async def feature_extraction_by_video(self, video_paths, out_dir=None):
        print(f"feature_extraction_by_video: video_paths: {video_paths}")
        if os.name == "nt":
            command = r"D:\Programs\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
        else:
            command = "FeatureExtraction"
        video_args = []
        feature_files = []
        for video_path in video_paths:
            video_args.append("-f")
            video_args.append(video_path)
            video_name = Path(video_path).name
            video_stem = Path(video_path).stem

            if not out_dir:
                video_directory = Path(video_path).parent
                out_dir = video_directory / "processed"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_dir = out_dir.as_posix()

            feature_file = f"{out_dir}/{video_stem}.csv"
            feature_files.append(feature_file)
        feature_args = ["-3Dfp", "-pose", "-aus", "-gaze"]
        out_dir_args = ["-out_dir", out_dir]
        args = video_args + feature_args + out_dir_args

        print(f"command: {command} , args: {args}")

        try:
            result = subprocess.Popen([command] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # 获取实时输出
            for line in iter(result.stdout.readline, b''):
                sys.stdout.write(line.decode('utf-8'))
                sys.stdout.flush()

            # 等待命令执行完成
            result.wait()
            return feature_files
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            print(e.output)
            return []

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
            result = subprocess.Popen([command] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # 获取实时输出
            for line in iter(result.stdout.readline, b''):
                sys.stdout.write(line.decode('utf-8'))
                sys.stdout.flush()

            # 等待命令执行完成
            result.wait()
            return feature_files
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            print(e.output)
            return []


openface_service = OpenFaceService()

