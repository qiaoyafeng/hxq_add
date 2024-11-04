import os
import subprocess
from pathlib import Path
import cv2
import uuid
import base64
import re
import shutil
import requests

from config import Config

TEMP_PATH = Config.get_temp_path()


def base64_encode(stream):
    return str(base64.b64encode(stream), "utf-8")


def base64_decode(base64_str: str):
    return base64.b64decode(base64_str)


def encode_file_to_base64(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def get_resp():
    resp = {
        "code": 0,
        "message": "操作成功！",
        "success": True,
        "data": {},
    }
    return resp


def build_resp(code, data, message=None):
    resp = {
        "code": code,
        "message": message,
        "success": code == 0,
        "data": data,
    }
    return resp


def safe_int(value, default=0):
    try:
        value = str(value).replace(",", "").split(".")[0].strip()
        return int(value)
    except Exception as e:
        _ = e
        return default


def get_files_by_ext(directory, extension):
    directory = Path(directory)
    print(f"directory: {directory}")

    files = list(directory.rglob("*" + extension))
    return files


def extract_visual_feature(video_path):
    print(f"extract_visual_feature: {video_path} ......")

    if os.name == "nt":
        feature_extraction_command = r"D:\Programs\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
    else:
        feature_extraction_command = "FeatureExtraction"

    args = ["-f", video_path]

    # 执行exe程序并传递参数
    try:
        result = subprocess.run([feature_extraction_command] + args, check=True, capture_output=True, text=True)
        # 打印输出结果
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(e.output)


def extract_visual_feature_by_images(image_paths: list, out_dir):
    print(f"extract_visual_feature_by_images: {image_paths} ......")

    if os.name == "nt":
        feature_extraction_command = r"D:\Programs\OpenFace_2.2.0_win_x64\FaceLandmarkImg.exe"
    else:
        feature_extraction_command = "FaceLandmarkImg"

    image_args = []
    for image_path in image_paths:
        image_args.append("-f")
        image_args.append(image_path)

    feature_args = ["-3Dfp", "-pose", "-aus", "-gaze"]
    out_dir_args = ["-out_dir", out_dir]

    args = image_args + feature_args + out_dir_args

    try:
        result = subprocess.run([feature_extraction_command] + args, check=True, capture_output=True, text=True)
        # 打印输出结果
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(e.output)


def replace_special_character(raw_str):
    update_str = (
        raw_str.replace("\n", "。")
        .replace("\r", "")
        .replace("【", "")
        .replace("】", "")
        .replace("[", "")
        .replace("]", "")
    )
    return update_str


async def download_file(file_url, file_path):
    with open(file_path, "wb") as file:
        r = requests.get(file_url)
        file.write(r.content)


def copy_file(source_path, target_path):
    shutil.copy(source_path, target_path)


def save_nth_frame(video_path, nth=1, suffix="jpg", image_dir=None):
    image_name, image_path = None, None
    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if nth > frame_count:
        nth = frame_count
    capture.set(cv2.CAP_PROP_POS_FRAMES, nth)
    success, image = capture.read()
    if success:
        image_name = f"{uuid.uuid4().hex}.{suffix}"
        if image_dir:
            image_path = f"{image_dir}/{image_name}"
        else:
            image_path = f"{TEMP_PATH}/{image_name}"
        cv2.imwrite(image_path, image)
    return image_name, image_path
