import json
import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import uvicorn

from fastapi.openapi.docs import get_swagger_ui_html

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, Request, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse
from mimetypes import guess_type

from api.detect import detect_api
from api.file import file_api
from api.schemas import ImageDetectRequest
from utils import get_resp, replace_special_character, build_resp, init_seed
from config import Config, settings

app = FastAPI(
    title="HXQ ADE",
    summary="HXQ Automatic Depression Detection",
    docs_url=None,
    redoc_url=None,
)

executor = ThreadPoolExecutor(10)

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

TEMP_PATH = Config.get_temp_path()


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/js/swagger-ui-bundle.js",
        swagger_css_url="/static/js/swagger-ui.css",
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    image_list = []

    return templates.TemplateResponse(
        "index.html", {"request": request, "image_list": image_list}
    )


@app.get("/detect", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("detect.html", {"request": request})


@app.post("/uploadfile/")
async def uploadfile(file: UploadFile):
    resp = get_resp()
    url, path, name = await file_api.uploadfile(file)
    file_info = {"path": path, "url": url, "name": name}
    resp["data"] = {"file_info": file_info}
    return resp


@app.get("/get_file/{file_name}", summary="Get file by file name")
def get_file(file_name: str):
    file_path = os.path.isfile(os.path.join(TEMP_PATH, file_name))
    if file_path:
        return FileResponse(os.path.join(TEMP_PATH, file_name))
    else:
        return {"code": 404, "message": "file does not exist."}


@app.post("/image/batch_upload", summary="图片批量上传")
async def image_batch_upload(
    batch_no: str = Form(),
    upload_images: list[UploadFile] = File(description="Multiple images as UploadFile"),
):
    image_paths = []
    print(f"batch_no: {batch_no}")
    for image in upload_images:
        dir_path = Path(f"{TEMP_PATH}/img/{batch_no}")
        dir_path.mkdir(parents=True, exist_ok=True)
        url, path, name = await file_api.uploadfile(image, dir_path)
        image_paths.append(path)
    return build_resp(0, {})


@app.post("/image/detect", summary="图片检测")
async def image_detect(
    batch_no: str = Form(),
    image1: UploadFile = File(),
    image2: UploadFile = File(),
    image3: UploadFile = File(),
):
    image_paths = []
    print(f"batch_no: {batch_no}")
    for image in [image1, image2, image3]:
        dir_path = Path(f"{TEMP_PATH}/img/{batch_no}")
        dir_path.mkdir(parents=True, exist_ok=True)
        url, path, name = await file_api.uploadfile(image, dir_path)
        image_paths.append(path)

    detect_dict = await detect_api.image_detect(image_paths, batch_no)
    return build_resp(0, {"detect": detect_dict})


@app.post("/image/batch_detect", summary="图片批量检测")
async def image_batch_detect(
    batch_no: str = Form(),
    files: list[UploadFile] = File(description="Multiple images as UploadFile"),
    is_multi_class: int = Form(),
):
    image_paths = []
    print(f"batch_no: {batch_no}")
    print(f"is_multi_class: {is_multi_class}")
    for image in files:
        dir_path = Path(f"{TEMP_PATH}/img/{batch_no}")
        dir_path.mkdir(parents=True, exist_ok=True)
        url, path, name = await file_api.uploadfile(image, dir_path)
        image_paths.append(path)

    detect_dict = await detect_api.image_detect(
        image_paths, batch_no, is_multi_class=is_multi_class
    )
    return build_resp(0, {"detect": detect_dict})


@app.post("/video/detect", summary="视频检测")
async def video_detect(
    batch_no: str = Form(None),
    video: UploadFile = File(),
):
    print(f"batch_no: {batch_no}")
    if not batch_no:
        batch_no = f"{uuid.uuid4().hex}"
    dir_path = Path(f"{TEMP_PATH}/video/{batch_no}")
    dir_path.mkdir(parents=True, exist_ok=True)
    url, path, name = await file_api.uploadfile(video, dir_path)
    detect_dict = await detect_api.video_detect(path, batch_no)
    return build_resp(0, {"batch_no": batch_no, "detect": detect_dict})


# 原来VUE格式的接口
@app.post("/video", summary="视频检测(原来VUE格式的接口)")
async def vue_video_detect(
    file: UploadFile = File(),
):
    batch_no = f"{uuid.uuid4().hex}"
    dir_path = Path(f"{TEMP_PATH}/video/{batch_no}")
    dir_path.mkdir(parents=True, exist_ok=True)
    url, path, name = await file_api.uploadfile(file, dir_path)
    detect_dict = await detect_api.video_detect(path, batch_no)
    depressed_index_str = detect_dict["depressed_index"]
    video_score = int(float(depressed_index_str.strip('%')))
    data = {'code': 200, 'data': int(video_score), 'msg': 'Video success'}
    return json.dumps(data)


if __name__ == "__main__":
    uvicorn.run(
        app="__main__:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.RELOAD,
    )
