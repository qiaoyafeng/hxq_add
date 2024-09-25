from service.detect import detect_service
from service.log import logger


class DetectAPI:
    def __init__(self):
        self.detect_service = detect_service

    async def image_detect(self, image_paths, batch_no):
        detect_dict = await self.detect_service.image_detect(image_paths, batch_no)
        return detect_dict


detect_api = DetectAPI()
