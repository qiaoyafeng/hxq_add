import asyncio
from pathlib import Path

from common.utils import save_nth_frame
from config import Config
from service.detect import detect_service
from service.log import logger
from service.common import notice_service

TEMP_PATH = Config.get_temp_path()


class DetectAPI:
    def __init__(self):
        self.detect_service = detect_service
        self.notice_service = notice_service
        self.video_detect_job_flag = False

    async def image_detect(self, image_paths, batch_no, is_multi_class=0):
        if is_multi_class:
            detect_dict = await self.detect_service.multi_class_detect(
                image_paths, batch_no
            )
        else:
            detect_dict = await self.detect_service.image_detect(image_paths, batch_no)
        return detect_dict

    async def video_detect(self, video_path, batch_no):
        detect_dict = await self.detect_service.video_detect_v2(video_path, batch_no)
        return detect_dict

    async def create_video_detect_task(self, video, batch_no):
        task = await self.detect_service.create_video_detect_task(video, batch_no)
        return task

    async def get_video_detect_task_by_batch_no(self, batch_no):
        task = self.detect_service.get_video_detect_task_by_batch_no(batch_no)
        return task

    async def get_all_video_detect_tasks(self):
        tasks = self.detect_service.get_all_video_detect_tasks()
        return tasks

    def video_detect_job(self):
        if self.video_detect_job_flag:
            return
        self.video_detect_job_flag = True
        tasks = self.detect_service.get_video_detect_task_by_step(0)
        # print(f"tasks: {tasks}")
        if len(tasks) == 0:
            self.video_detect_job_flag = False
            return
        for task in tasks:
            if not task["del_status"]:
                video_name = task["video"]
                batch_no = task["batch_no"]
                video_path = Path(f"{TEMP_PATH}/video/{batch_no}/{video_name}")
                if batch_no and video_name:
                    logger.info(f"开始执行视频检测定时任务: {task}")
                    try:
                        result = asyncio.run(
                            self.detect_service.video_detect_v2(video_path, batch_no)
                        )
                        print(f"result: {result}")
                        data_dict = {
                            "id": task["id"],
                            "point": result["point"],
                            "diagnosis": result["diagnosis"],
                            "depressed_score": result["depressed_score"],
                            "depressed_state": result["depressed_state"],
                            "depressed_score_list": result["depressed_score_list"],
                            "current_step": 3,
                        }
                    except Exception as e:
                        print(f"执行视频检测定时任务{task}异常: {e}")
                        data_dict = {
                            "id": task["id"],
                            "point": 0,
                            "diagnosis": "F32",
                            "depressed_score": 0,
                            "depressed_state": "正常",
                            "depressed_score_list": "",
                            "current_step": 3,
                            "comment": "任务处理异常",
                        }

                    self.detect_service.udpate_video_detect_task(data_dict)

        self.video_detect_job_flag = False

    def create_video_detect_cover_image_job(self):
        tasks = self.detect_service.get_all_video_detect_tasks()
        for task in tasks:
            if not task["cover_image"] and not task["del_status"]:
                logger.info(f"开始执行视频检测任务缩略图定时任务: {task}")
                batch_no_dir = Path(f"{TEMP_PATH}/video/{task['batch_no']}")
                video_path = batch_no_dir / task["video"]
                # 获取视频第一帧作为任务封面
                first_frame_name, first_frame_path = save_nth_frame(
                    video_path, image_dir=batch_no_dir
                )
                data_dict = {
                    "id": task["id"],
                    "cover_image": first_frame_name,
                }
                self.detect_service.udpate_video_detect_task(data_dict)

    async def send_sms_job(self):
        unsent_sms_flag = 0
        sent_sms_success_flag = 1
        sent_sms_fail_flag = 1
        task_complete_flag = 3
        unsent_sms_tasks = self.detect_service.get_video_detect_task_by_sms_status(unsent_sms_flag)
        for task in unsent_sms_tasks:
            if not task["del_status"] and task["current_step"] == task_complete_flag:
                task_id = task["id"]
                phone = task["phone"]
                batch_no = task["batch_no"]
                create_time = task["create_time"].to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"开始执行发送短信通知任务: task_id: {task_id}, phone: {phone}, batch_no: {batch_no}, create_time: {create_time}")
                info = {"task_id": task_id, "batch_no": batch_no, "create_time": create_time}
                send_res = await self.notice_service.sms(phone, info)
                if send_res["code"] == 200:
                    sms_status = sent_sms_success_flag
                else:
                    sms_status = sent_sms_fail_flag
                data_dict = {
                    "id": task["id"],
                    "sms_status": sms_status,
                }
                self.detect_service.udpate_video_detect_task(data_dict)

    async def bind_phone_to_task(self, task_id, phone):
        task = await self.detect_service.bind_phone_to_task(task_id, phone)
        return task


detect_api = DetectAPI()
