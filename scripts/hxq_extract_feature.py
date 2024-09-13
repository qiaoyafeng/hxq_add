from pathlib import Path

from common.utils import extract_visual_feature, get_files_by_ext


def batch_extract_visual_feature(input_directory, extension="mp4"):
    files = get_files_by_ext(input_directory, extension)
    print(f"files:{files}")
    for file_path in files:
        extract_visual_feature(file_path.as_posix())


if __name__ == "__main__":
    subclip_doctor = r"E:\myworkspace\hxq_ade\datasets\hxq\videos_clip_subclip_doctor"
    # batch_extract_visual_feature(subclip_doctor)

    subclip_patient = r"E:\myworkspace\hxq_ade\datasets\hxq\videos_clip_subclip"
    batch_extract_visual_feature(subclip_patient)
