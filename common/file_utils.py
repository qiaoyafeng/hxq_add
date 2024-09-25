import os.path
import sys
import json
import random
from fastapi import UploadFile
import shutil
import zipfile
import re


def write_line(obj, file_path, file_name, overwrite=False):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    path = file_path + "/" + file_name
    if overwrite and os.path.exists(path):
        os.remove(path)
    with open(path, '+a', encoding='utf-8') as f:
        for data in obj:
            f.writelines(data + "\n")
    f.close()


def zip_folder_file(folder_path, saveFileName):
    if os.path.exists(saveFileName):
        os.remove(saveFileName)

    zipf = zipfile.ZipFile(saveFileName, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
    # # 创建zip文件并设置密码
    # with zipfile.ZipFile(saveFileName, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
    #     # 遍历文件夹中的所有文件和子文件夹
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isdir(file_path):
            continue
        # 将相对路径转换为绝对路径，并将其添加到zip文件中
        abs_file_path = os.path.abspath(file_path)
        zipf.write(abs_file_path, arcname=file_path[len(folder_path):])

    zipf.close()


def save(file, file_type, tags):
    for tag in tags:
        dataset_dir = "dataset"
        if file_type == 1:
            dataset_dir = os.path.join(dataset_dir, "img", tag)
        else:
            dataset_dir = os.path.join(dataset_dir, "audio", tag)

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        file.save(os.path.join(dataset_dir, file.filename))


def save_up(file, file_type, dataset_dir, temp_path=None):
    if file_type == 1:
        dataset_dir = f"{dataset_dir}/img_pre"
    else:
        dataset_dir = f"{dataset_dir}/video"

    if not temp_path:
        temp_path = str(random.randint(1000000, 2000000))
    else:
        temp_path = f"pic_{temp_path}"
    dataset_dir = f"{dataset_dir}/{temp_path}"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    video_name = "up_" + (chinese_pattern.sub('c', file.filename))
    file.save(os.path.join(dataset_dir, video_name))
    return temp_path, dataset_dir, video_name


def save_up_file(file: UploadFile, file_type, dataset_dir, temp_path=None):
    if file_type == 1:
        dataset_dir = f"{dataset_dir}/img_pre"
    else:
        dataset_dir = f"{dataset_dir}/video"

    if not temp_path:
        temp_path = str(random.randint(1000000, 2000000))
    else:
        temp_path = f"pic_{temp_path}"
    dataset_dir = f"{dataset_dir}/{temp_path}"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    video_name = "up_" + file.filename
    with open(os.path.join(dataset_dir, video_name), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path, dataset_dir, video_name


def child_files_count(base_path):
    file_count = {}
    for filename in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, filename)):
            file_count[filename] = 0
            for chilname in os.listdir(os.path.join(base_path, filename)):
                file_count[filename] = file_c(os.path.join(base_path, filename))

    with open(base_path + "/" + 'file_count.txt', 'w', encoding='utf-8') as f:
        f.write(json.dumps(file_count))


def child_files(base_path):
    file_name = []
    for filename in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, filename)):
            file_name += child_files(f'{base_path}/{filename}')
        else:
            file_name.append(f'{base_path}/{filename}')
    return file_name


def child_file_fold(base_path):
    file_name = []
    for filename in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, filename)):
            childf = child_file_fold(f'{base_path}/{filename}')
            for cf in childf:
                file_name.append(cf)
        else:
            if base_path not in file_name:
                file_name.append(base_path)
    return file_name


def file_c(base_path):
    count = 0
    for filename in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, filename)):
            count += file_c(os.path.join(base_path, filename))
        else:
            count += 1
    return count


def read_file(path, split=","):
    data = []
    if not os.path.exists(path):
        raise ValueError(f"{path} 文件不存在")
    with open(path, encoding='utf-8') as f:
        line = f.readline()
        while line:
            data.append(line.replace("\n", "").replace("\\n", "\n").split(split))
            line = f.readline()
    f.close()
    return data


def read_file_all(path):
    data = []
    if not os.path.exists(path):
        raise ValueError(f"{path} 文件不存在")
    with open(path, encoding='utf-8') as f:
        line = f.readline()
        while line:
            data.append(line.replace("\n", ""))
            line = f.readline()
    f.close()
    return data


def file_stream(path):
    """
    读取文件流内容
    :param path: 文件路径
    :return:
    """
    with open(path, 'rb') as file:
        file_content = file.read()
    return file_content


def cp_file(src, dest):
    print(f"copy {src} to {dest}")
    shutil.copyfile(src, dest)


def empty_folder(folder):
    empty_folders = []
    for root, dirs, files in os.walk(folder):
        if not dirs and not files:
            empty_folders.append(root)
    return empty_folders


def remove_folder(path):
    if os.path.exists(path):
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        else:
            for filename in os.listdir(path):
                remove_folder(os.path.join(path, filename))
            os.rmdir(path)


def replace_n(path):
    data = []
    with open(path, encoding='utf-8') as f:
        line = f.readline()
        while line:
            for li in line.replace("\n", "").split("/n"):
                data.append(li)
            line = f.readline()
    f.close()

    os.remove(path)
    with open(path, '+a', encoding='utf-8') as f:
        for d in data:
            if len(d) > 0:
                f.writelines(d + "\n")


if __name__ == '__main__':
    # print(child_files_count("/opt/emotion/file/video"))
    # print(empty_folder("/opt/emotion/file/label/img_pre"))
    # rootpath = "/opt/emotion/data/emo"
    rootpath = "/opt/emotion/file/img_pre"
    mon = '2023-08-01'
    total = 0
    for dia in os.listdir(rootpath):
        if os.path.exists(rootpath + "/" + dia + "/" + mon):
            for video in os.listdir(rootpath + "/" + dia + "/" + mon):
                print(video)

    # 替换换行
    # for dia in os.listdir(rootpath):
    #     for video in os.listdir(rootpath + "/" + dia):
    #         print(video)
    #         if video.endswith(".mp4.csv"):
    #             replace_n(f"{rootpath}/{dia}/{video}")

    # 删除空文件夹
    # emp = empty_folder(rootpath)
    # for e in emp:
    #     remove_folder(e)

    sys.exit()
