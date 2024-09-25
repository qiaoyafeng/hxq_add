from fastapi import UploadFile

from config import Config
from service.file import file_service

TEMP_PATH = Config.get_temp_path()


class FileAPI:

    def __init__(self):
        self.file_service = file_service

    async def save(self):
        file_url, file_path, file_name = await self.file_service.save_file()
        return file_url, file_path, file_name

    async def uploadfile(self, file: UploadFile, dir_path=None):
        file_url, file_path, file_name = await self.file_service.uploadfile(file, dir_path=dir_path)
        return file_url, file_path, file_name


file_api = FileAPI()
