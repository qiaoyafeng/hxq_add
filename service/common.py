import json
import uuid

import requests
from fastapi import UploadFile

from config import settings, Config
from common.utils import download_file, copy_file


class NoticeService:
    def __init__(self):
        self.headers = {"Content-Type": "application/x-www-form-urlencoded"}
        self.sms_url = f"{settings.SMS_HOST}/faceResult/sms"

    # 短信通知
    async def sms(self, phone, info):
        data = {"mobile": phone, "time": info["create_time"].split(" ")[0], "batch": info["batch_no"]}
        print(f"send_sms data: {data}")
        res = requests.post(url=self.sms_url, data=data, headers=self.headers)
        r_dict = res.json()
        print(f"send_sms r_dict: {r_dict}")
        return r_dict


notice_service = NoticeService()
