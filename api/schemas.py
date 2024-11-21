import datetime
from typing import List, Union, Optional
from fastapi import File, Form, UploadFile

from pydantic import BaseModel


class ImageDetectRequest(BaseModel):
    batch_no: str
    image1: bytes = File()
    image2: bytes = File()
    image3: bytes = File()


class BindPhoneRequest(BaseModel):
    batch_no: str
    phone: str
