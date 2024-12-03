import base64
import random
import re
import shutil

import librosa
import numpy as np
import requests
import torch


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


def init_seed(manual_seed=1):
    """
    Set random seed for torch and numpy.
    """
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_audio_duration(file_path):
    duration = librosa.get_duration(path=file_path)
    return duration