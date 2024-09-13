import uuid

from fastapi import UploadFile

from config import settings, Config
from common.utils import download_file, copy_file

TEMP_PATH = Config.get_temp_path()


class DataService:
    def __init__(self):
        pass





data_service = DataService()
