import os
from pathlib import Path

from moviepy.editor import VideoFileClip

from common.utils import get_files_by_ext


def subclip(input_video_path, output_video_path, fps=30, start=0, end=0):
    """截切视频片段"""

    # 加载视频文件
    video = VideoFileClip(input_video_path)

    # 获取视频的宽度和高度
    width, height = video.size
    duration = video.duration
    if start > duration:
        start = 0
    if end > duration or end <= 0:
        end = duration

    sub_video = video.subclip(start, end)

    # 保存裁剪后的视频
    sub_video.write_videofile(output_video_path, fps=fps, codec="libx264", audio=False)


def clip_video(input_video_path, output_video_path, fps=30, role="patient"):
    """

    :param input_video_path:
    :param output_video_path:
    :param fps:
    :param role: patient / doctor
    :return:
    """

    # 加载视频文件
    video = VideoFileClip(input_video_path)

    # 获取视频的宽度和高度
    width, height = video.size

    # 裁剪左半部分的视频 (从左上角到宽度的一半)
    if role == "patient":
        half_video = video.crop(x1=0, y1=0, x2=width / 2, y2=height)
    else:
        half_video = video.crop(x1=width / 2, y1=0, x2=width, y2=height)

    # 保存裁剪后的视频
    half_video.write_videofile(output_video_path, fps=fps, codec="libx264", audio=False)


def batch_clip_videos(
    input_directory, output_directory, extension="mp4", role="patient"
):
    os.makedirs(output_directory, exist_ok=True)
    files = get_files_by_ext(input_directory, extension)
    print(f"files:{files}")
    for file_path in files:
        if role != "patient":
            file_name = f"{file_path.stem}_normal{file_path.suffix}"
        else:
            file_name = file_path.name
        output_video_path = Path(output_directory) / file_name
        clip_video(file_path.as_posix(), output_video_path.as_posix(), role=role)


def batch_subclip_videos(input_directory, output_directory, extension="mp4", start=60, end=240):
    os.makedirs(output_directory, exist_ok=True)
    files = get_files_by_ext(input_directory, extension)
    print(f"files:{files}")
    for file_path in files:
        output_video_path = Path(output_directory) / file_path.name
        subclip(
            file_path.as_posix(), output_video_path.as_posix(), fps=30, start=start, end=end
        )


if __name__ == "__main__":
    input_file = "10.mp4"
    output_file = "10_clip.mp4"
    # clip_video(input_file, output_file)

    input_directory = r"E:\myworkspace\hxq_ade\data\hxq\video"

    subclip_directory = (
            r"E:\myworkspace\hxq_ade\data\hxq\video_subclip"
        )

    # batch_subclip_videos(input_directory, subclip_directory,  start=60, end=180)

    output_directory_patient = r"E:\myworkspace\hxq_ade\data\hxq\video_subclip_patient"
    output_directory_doctor = r"E:\myworkspace\hxq_ade\data\hxq\video_subclip_doctor"

    batch_clip_videos(subclip_directory, output_directory_patient)
    batch_clip_videos(subclip_directory, output_directory_doctor, role="doctor")

    output_directory_subclip = (
        r"E:\myworkspace\hxq_ade\data\hxq\videos_clip_subclip"
    )
    # batch_subclip_videos(output_directory, output_directory_subclip)

    output_directory_subclip_doctor = (
        r"E:\myworkspace\hxq_ade\data\hxq\videos_clip_subclip_doctor"
    )
    # batch_subclip_videos(output_directory_doctor, output_directory_subclip_doctor)
