import os
from pydantic_settings import BaseSettings
from functools import lru_cache

config_path = __file__


class Config:
    @classmethod
    def get_home_path(cls):
        return os.path.dirname(config_path)

    @classmethod
    def check_path(cls, dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    @classmethod
    def get_temp_path(cls):
        temp_path = os.path.join(cls.get_home_path(), "temp")
        cls.check_path(temp_path)
        return temp_path

    @classmethod
    def get_log_path(cls):
        log_path = os.path.join(cls.get_home_path(), "log")
        cls.check_path(log_path)
        return log_path

    @classmethod
    def get_checkpoints_path(cls):
        checkpoints_path = os.path.join(cls.get_home_path(), "checkpoints")
        cls.check_path(checkpoints_path)
        return checkpoints_path


class Settings(BaseSettings):
    BASE_DOMAIN: str = "http://127.0.0.1:8000"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    WORKERS: int = 1
    RELOAD: bool = False

    # DB
    DB_IP: str = "127.0.0.1"
    DB_PORT: int = 3306
    DB_NAME: str = "hxq_ade"
    DB_USERNAME: str = "root"
    DB_PASSWORD: str = "123456"

    # SCHEDULER
    IS_SCHEDULER: bool = True

    # MODEL
    MODEL_WEIGHTS_PATH: str = "weights/binary_model.pt"
    MODEL_MULTI_CLASS_WEIGHTS_PATH: str = "weights/multi_class_model.pt"
    MULTI_CLASS_METHOD: str = "one2one"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
