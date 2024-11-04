import uuid

from fastapi import UploadFile

from config import settings, Config
from common.utils import download_file, copy_file

TEMP_PATH = Config.get_temp_path()


class FileService:
    def __init__(self):
        pass

    # 根据URL保存文件
    async def save_file(self, file_url: str, suffix="wav"):
        file_name = f"{uuid.uuid4().hex}.{suffix}"
        file_path = f"{TEMP_PATH}/{file_name}"
        await download_file(file_url, file_path)
        local_url = f"{settings.BASE_DOMAIN}/get_file/{file_name}"
        return local_url, file_path, file_name

    # 上传文件
    async def uploadfile(self, file: UploadFile, dir_path=None):
        suffix = file.filename.split(".")[-1]
        if suffix == "blob":
            suffix = "mp4"
        file_name = f"{uuid.uuid4().hex}.{suffix}"
        file_path = f"{dir_path}/{file_name}" if dir_path else f"{TEMP_PATH}/{file_name}"
        local_url = f"{settings.BASE_DOMAIN}/get_file/{file_name}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return local_url, file_path, file_name


file_service = FileService()
