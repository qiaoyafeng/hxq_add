import json
import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from fastapi.openapi.docs import get_swagger_ui_html

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, Request, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse
from mimetypes import guess_type
from apscheduler.schedulers.background import BackgroundScheduler

from api.detect import detect_api
from api.file import file_api
from api.schemas import ImageDetectRequest, BindPhoneRequest
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

scheduler = BackgroundScheduler()
async_scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def startup_event():
    scheduler.start()
    async_scheduler.start()
    if settings.IS_SCHEDULER:
        scheduler.add_job(detect_api.video_detect_job, "interval", seconds=5)
        scheduler.add_job(
            detect_api.create_video_detect_cover_image_job, "interval", seconds=20
        )
        scheduler.add_job(
            detect_api.audio_detect_job, "interval", seconds=20
        )

    if settings.IS_SEND_SMS:
        async_scheduler.add_job(
            detect_api.send_sms_job, "interval", seconds=10
        )


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
@app.post("/login", summary="登录(原来VUE格式的接口)")
async def vue_login_post(username: str = Form(), password: str = Form()):
    print(username)
    print("password" + password)
    if username == "" or password == "":
        print("用户名和密码不能为空！！！")
        data = {"code": 401, "data": "", "msg": "username or password can not null"}
        return data

    print("login success")
    data = {"code": 200, "data": 1000, "msg": "login success"}
    return data


@app.get("/Time", summary="Get Time")
def get_time(timeBegin: int = 0, timeEnd: int = 0):
    print(f"timeBegin: {timeBegin}")
    print(f"timeEnd: {timeEnd}")
    data = {"code": 200, "data": "", "msg": "Time success"}
    return data


@app.post("/video", summary="视频检测(原来VUE格式的接口)")
async def vue_video_detect(
    file: UploadFile = File(),
):
    print(f"vue_video_detect: file:{file}")
    batch_no = f"{uuid.uuid4().hex}"
    dir_path = Path(f"{TEMP_PATH}/video/{batch_no}")
    dir_path.mkdir(parents=True, exist_ok=True)
    url, path, name = await file_api.uploadfile(file, dir_path)
    detect_dict = await detect_api.video_detect(path, batch_no)
    print(f"vue_video_detect: {detect_dict}")
    return build_resp(0, {"batch_no": batch_no, "detect": detect_dict})


@app.post("/api/create_batch_no", summary="创建批次号")
async def create_batch_no():
    print(f"create_video_detect_task")
    batch_no = f"{uuid.uuid4().hex}"
    batch_no = batch_no[:10]
    return build_resp(0, {"batch_no": batch_no}, message="create batch_no success")


@app.post("/api/create_video_detect_task", summary="创建视频检测任务")
async def create_video_detect_task(
    file: UploadFile = File(),
):
    print(f"create_video_detect_task: file:{file}")
    batch_no = f"{uuid.uuid4().hex}"
    batch_no = batch_no[:10]
    dir_path = Path(f"{TEMP_PATH}/video/{batch_no}")
    dir_path.mkdir(parents=True, exist_ok=True)
    url, path, name = await file_api.uploadfile(file, dir_path)
    await detect_api.create_video_detect_task(name, batch_no)
    return build_resp(0, {"batch_no": batch_no}, message="create task success")


@app.get("/api/get_video_detect_task", summary="获取视频检测任务")
async def get_video_detect_task(
    batch_no: str,
):
    print(f"get_video_detect_task: task_id:{batch_no}")
    task = await detect_api.get_video_detect_task_by_batch_no(batch_no)
    is_completed = 0
    if task:
        print(f"get_video_detect_task task: {task}")
        if task["current_step"] == 3:
            is_completed = 1
            task_info = {
                "main_disease": {
                    "diagnosis": task["diagnosis"],
                    "point": task["point"],
                    "name": "抑郁性障碍",
                },
                "detect": {
                    "depressed_id": 0,
                    "depressed_state": task["diagnosis"],
                    "depressed_index": task["depressed_score"],
                    "depressed_score": task["depressed_score"],
                    "depressed_score_list": task["depressed_score_list"],
                },
                "detect_list": [],
            }
        else:
            task_info = {}
    else:
        task_info = {}

    return build_resp(
        0,
        {"batch_no": batch_no, "is_completed": is_completed, "task_info": task_info},
        message="get task success",
    )


@app.get(
    "/api/get_task_file/{batch_no}/{file_name}", summary="Get task file by file name"
)
def get_task_file(batch_no: str, file_name: str):
    file_path = os.path.isfile(os.path.join(TEMP_PATH, "video", batch_no, file_name))
    if file_path:
        return FileResponse(os.path.join(TEMP_PATH, "video", batch_no, file_name))
    else:
        return {"code": 404, "message": "file does not exist."}


@app.get("/api/get_all_video_detect_tasks", summary="获取所有视频检测任务")
async def get_all_video_detect_tasks():
    print(f"get_all_video_detect_tasks")
    task_list = []
    tasks = await detect_api.get_all_video_detect_tasks()
    for task in tasks:
        if not task["del_status"]:
            task_dict = {
                "task_id": task["id"],
                "batch_no": task["batch_no"],
                "video": task["video"],
                "video_url": f"/api/get_task_file/{task['batch_no']}/{task['video']}",
                "point": task["point"],
                "diagnosis": task["diagnosis"],
                "cover_image": task["cover_image"],
                "cover_image_url": f"/api/get_task_file/{task['batch_no']}/{task['cover_image']}",
                "depressed_score": task["depressed_score"],
                "depressed_score_list": task["depressed_score_list"],
                "create_time": task["create_time"],
            }
            task_list.append(task_dict)

    return build_resp(
        0,
        {"task_list": task_list},
        message="get task success",
    )


@app.post("/api/bind_phone_to_task", summary="绑定手机号到任务")
async def bind_phone_to_task(request: BindPhoneRequest):
    batch_no = request.batch_no
    phone = request.phone
    print(f"bind_phone_to_task: batch_no:{batch_no}, phone: {phone}")

    task = await detect_api.get_video_detect_task_by_batch_no(batch_no)
    if task:
        task_id = task["id"]
        await detect_api.bind_phone_to_task(task_id, phone)
        return build_resp(
            0,
            {"task_id": task_id, "batch_no": batch_no, "phone": phone},
            message="bind success",
        )
    else:
        return build_resp(
            200,
            {"task_id": "", "batch_no": batch_no, "phone": phone},
            message="task does not exist.",
        )


@app.post("/api/create_audio_detect_task", summary="创建音频检测任务")
async def create_audio_detect_task(
    file: UploadFile = File(), batch_no: str = ""
):
    print(f"create_audio_detect_task: file:{file}")
    if not batch_no:
        batch_no = f"{uuid.uuid4().hex}"
        batch_no = batch_no[:10]
    dir_path = Path(f"{TEMP_PATH}/audio/{batch_no}")
    dir_path.mkdir(parents=True, exist_ok=True)
    url, path, name = await file_api.uploadfile(file, dir_path)
    await detect_api.create_audio_detect_task(name, batch_no)
    return build_resp(0, {"batch_no": batch_no}, message="create task success")


@app.get("/api/get_audio_detect_task", summary="获取音频检测任务")
async def get_audio_detect_task(
    batch_no: str,
):
    print(f"get_audio_detect_task: task_id:{batch_no}")
    task = await detect_api.get_audio_detect_task_by_batch_no(batch_no)
    is_completed = 0
    if task:
        print(f"get_audio_detect_task task: {task}")
        if task["current_step"] == 3:
            is_completed = 1
            task_info = {
                "main_disease": {
                    "diagnosis": task["diagnosis"],
                    "point": task["point"],
                    "name": "抑郁性障碍",
                },
                "detect": {
                    "depressed_id": 0,
                    "depressed_state": task["diagnosis"],
                    "depressed_index": task["depressed_score"],
                    "depressed_score": task["depressed_score"],
                    "depressed_score_list": task["depressed_score_list"],
                },
                "detect_list": [],
            }
        else:
            task_info = {}
    else:
        task_info = {}

    return build_resp(
        0,
        {"batch_no": batch_no, "is_completed": is_completed, "task_info": task_info},
        message="get audio task success",
    )

if __name__ == "__main__":
    init_seed()
    uvicorn.run(
        app="__main__:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.RELOAD,
    )
