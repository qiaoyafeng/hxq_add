import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from tcn import TCN
import os
import subprocess

import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from tcn import TCN



def HDR(video_new_path, HDR_path):
    m = [10, 20, 30, 40, 50]
    df = pd.read_csv(video_new_path)
    x0 = 5
    y0 = 73
    l = []
    for i in range(4080):
        l.append(i)
    l = np.array(l).reshape((1, len(l)))
    df1 = pd.DataFrame(l)
    for i in range(0, df.shape[0] - 101, 10):  # 每一帧
        lines = []
        print(i)
        for j in range(len(m)):  # 每一个时间间隔
            for k in range(0, 68):  # 每一对(x,y)
                a, b = i + m[j], i
                r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12 = (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                num = 0
                while a <= i + 100:
                    x = float(df.iloc[a][x0 + k] - df.iloc[b][x0 + k])
                    y = float(df.iloc[a][y0 + k] - df.iloc[b][y0 + k])
                    b = b + 10
                    a = b + m[j]
                    if x < -20:
                        r1 = r1 + 1
                    elif x >= -20 and x < -10:
                        r2 = r2 + 1
                    elif x >= -10 and x < 0:
                        r3 = r3 + 1
                    elif x >= 0 and x < 10:
                        r4 = r4 + 1
                    elif x >= 10 and x < 20:
                        r5 = r5 + 1
                    else:
                        r6 = r6 + 1
                    if y < -20:
                        r7 = r7 + 1
                    elif y >= -20 and y < -10:
                        r8 = r8 + 1
                    elif y >= -10 and y < 0:
                        r9 = r9 + 1
                    elif y >= 0 and y < 10:
                        r10 = r10 + 1
                    elif y >= 10 and y < 20:
                        r11 = r11 + 1
                    else:
                        r12 = r12 + 1
                    num = num + 1
                r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12 = (
                    r1 / num,
                    r2 / num,
                    r3 / num,
                    r4 / num,
                    r5 / num,
                    r6 / num,
                    r7 / num,
                    r8 / num,
                    r9 / num,
                    r10 / num,
                    r11 / num,
                    r12 / num,
                )
                lines.append(r1)
                lines.append(r2)
                lines.append(r3)
                lines.append(r4)
                lines.append(r5)
                lines.append(r6)
                lines.append(r7)
                lines.append(r8)
                lines.append(r9)
                lines.append(r10)
                lines.append(r11)
                lines.append(r12)
        lines = np.array(lines).reshape((1, len(lines)))
        df2 = pd.DataFrame(lines)
        df1 = pd.concat([df1, df2], ignore_index=True)
    df1 = df1[1:]
    df1.to_csv(HDR_path)


def gen_HDR(video_feature_path, hdr_path):
    m = [10, 20, 30, 40, 50]
    df = pd.read_csv(video_feature_path)
    x0 = 5
    y0 = 73
    l = []
    for i in range(4080):
        l.append(i)
    l = np.array(l).reshape((1, len(l)))
    df1 = pd.DataFrame(l)
    for i in range(0, df.shape[0] - 101, 10):  # 每一帧
        lines = []
        print(i)
        for j in range(len(m)):  # 每一个时间间隔
            for k in range(0, 68):  # 每一对(x,y)
                a, b = i + m[j], i
                r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12 = (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                num = 0
                while a <= i + 100:
                    x = float(df.iloc[a][x0 + k] - df.iloc[b][x0 + k])
                    y = float(df.iloc[a][y0 + k] - df.iloc[b][y0 + k])
                    b = b + 10
                    a = b + m[j]
                    if x < -20:
                        r1 = r1 + 1
                    elif x >= -20 and x < -10:
                        r2 = r2 + 1
                    elif x >= -10 and x < 0:
                        r3 = r3 + 1
                    elif x >= 0 and x < 10:
                        r4 = r4 + 1
                    elif x >= 10 and x < 20:
                        r5 = r5 + 1
                    else:
                        r6 = r6 + 1
                    if y < -20:
                        r7 = r7 + 1
                    elif y >= -20 and y < -10:
                        r8 = r8 + 1
                    elif y >= -10 and y < 0:
                        r9 = r9 + 1
                    elif y >= 0 and y < 10:
                        r10 = r10 + 1
                    elif y >= 10 and y < 20:
                        r11 = r11 + 1
                    else:
                        r12 = r12 + 1
                    num = num + 1
                r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12 = (
                    r1 / num,
                    r2 / num,
                    r3 / num,
                    r4 / num,
                    r5 / num,
                    r6 / num,
                    r7 / num,
                    r8 / num,
                    r9 / num,
                    r10 / num,
                    r11 / num,
                    r12 / num,
                )
                lines.append(r1)
                lines.append(r2)
                lines.append(r3)
                lines.append(r4)
                lines.append(r5)
                lines.append(r6)
                lines.append(r7)
                lines.append(r8)
                lines.append(r9)
                lines.append(r10)
                lines.append(r11)
                lines.append(r12)
        lines = np.array(lines).reshape((1, len(lines)))
        df2 = pd.DataFrame(lines)
        df1 = pd.concat([df1, df2], ignore_index=True)
    df1 = df1[1:]
    df1.to_csv(hdr_path)


def split_feature(feature_path, out_path, score_index):
    frame_size = 1800
    df = pd.read_csv(feature_path)
    df_len = df.shape[0]
    for i in range(int(df_len/frame_size)):
        frame_kp = df[i*frame_size:(i+1)*frame_size]
        out_file = f"{out_path}/{score_index}_{i}.csv"
        frame_kp.to_csv(out_file)


def get_files_by_ext(directory, extension):
    directory = Path(directory)
    print(f"directory: {directory}")

    files = list(directory.rglob("*" + extension))
    return files


if __name__ == "__main__":
    # video_name = "20241022143327"
    # video_path = f"F32/{video_name}.mp4"
    # video_feature(video_path)
    # video_new_path = f"E:/myworkspace/depression/depression/hxq_model/processed/{video_name}.csv"
    # HDR_path = f"{video_name}_video_capture.csv"
    # HDR(video_new_path, HDR_path)

    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24]

    root_database_path = r"E:\myworkspace\hxq_ade\data\hxq\multi_class_tcn"
    database_path = r"E:\myworkspace\hxq_ade\data\hxq\multi_class_tcn\F20"
    split_feature_dir = r"E:\myworkspace\hxq_ade\data\hxq\multi_class_tcn\F20_split_dir"

    train_dfs = []
    # for score in scores:
    #     print(f"score: {score}")
    #     score_dir = rf"{database_path}\{score}"
    #     score_files = get_files_by_ext(score_dir, ".csv")
    #     index = 0
    #     for score_file in score_files:
    #         score_index = f"{score}_{index}"
    #         split_feature(score_file, split_feature_dir, score_index)
    #         index += 1

    files = get_files_by_ext(split_feature_dir, ".csv")
    for file in files:
        file_name = os.path.basename(file)
        hdr_path = rf"{root_database_path}\F20_hdr\{file_name}"
        feature_data = gen_HDR(file, hdr_path)
        df = pd.read_csv(hdr_path, index_col=0)
        train_dfs.append(df)
    train_data = pd.concat(train_dfs, ignore_index=True)
    print(f"train_data: {train_data}")

