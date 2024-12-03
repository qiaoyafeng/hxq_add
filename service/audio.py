import os
import subprocess
from re import A
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tcn import TCN


# 通过openSmile提取音频特征
def audio_feature(audio_path, audio_feature_txt, start=0, end=-1):
    print(
        f"audio_Feature: audio_path:{audio_path},audio_feature_txt: {audio_feature_txt} , start: {start}, end: {end} ......"
    )
    start, end = str(start), str(end)
    if os.name == "nt":
        feature_command = r"D:/Programs/opensmile-3.0-win-x64/bin/SMILExtract.exe"
        args = [
            "-C",
            "D:/Programs/opensmile-3.0-win-x64/config/is09-13/IS09_emotion.conf",
            "-I",
            audio_path,
            "-O",
            audio_feature_txt,
            "-start",
            start,
            "-end",
            end,
        ]
    else:
        feature_command = "SMILExtract"
        args = [
            "-C",
            "",
            "-I",
            audio_path,
        ]

    # 执行exe程序并传递参数
    try:
        result = subprocess.run(
            [feature_command] + args, check=True, capture_output=True, text=True
        )
        # 打印输出结果
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(e.output)


# 将openSmile提取出txt文件转换成csv文件
def audio_feature_txt2csv(audio_feature_txt, audio_feature_csv):
    with open(audio_feature_txt, "r", encoding="utf8") as f:
        lines = f.readlines()
        s = lines[391].split(",")
        s = s[1:-1]
        s = np.array(s).reshape((1, len(s)))
        df = pd.DataFrame(s)
        for i in range(392, len(lines)):
            # print(i)
            t = lines[i].split(",")
            t = t[1:-1]
            t = np.array(t).reshape((1, len(t)))
            t = pd.DataFrame(t)
            df = pd.concat([df, t], ignore_index=True)
        df.to_csv(audio_feature_csv)


# 将音频特征送入模型中的出结果
def infer_audio_model(audio_feature_csv, weight_file="weights/audio_1.h5"):
    tcn = tf.keras.models.load_model(
        weight_file, custom_objects={"TCN": TCN, "mse": "mse"}
    )
    df = pd.read_csv(audio_feature_csv, index_col=0)
    x_test = df.to_numpy()
    print(f"infer_video_model x_test shape: {x_test.shape}")
    x_test_normal = x_test.reshape(-1, df.shape[1])
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x_test_minmax = min_max_scaler.fit_transform(x_test_normal)
    x_test = x_test_minmax.reshape(df.shape[0], -1, df.shape[1])
    predict = tcn.predict(x_test)
    predict_list = []
    for i in range(len(predict)):
        predict_list.append(predict[i][0])
    min_x = np.min(predict)
    print(f"infer_audio_model: min_x: {min_x}, predict_list: {predict_list}")
    return min_x, predict_list


if __name__ == "__main__":
    audio_path = r"E:\myworkspace\hxq_ade\temp\audio\aa8afc70f9\7bca7433349f4203a894e76cb9e7db54.wav"
    audio_path = r"E:\myworkspace\hxq_ade\temp\audio\c104604ce6\3600763039e94c71838ff04cbc79c2a5.mp3.wav"
    audio_feature_txt = "audio_feature_test.txt"
    audio_feature(audio_path, audio_feature_txt, start=0, end=10)
    audio_feature_csv = r"E:\myworkspace\hxq_ade\temp\audio\1658c1e6e5\audio_feature.csv"
    audio_feature_csv = r"E:\myworkspace\hxq_ade\temp\audio\a2f65ef9fc\audio_feature.csv"
    # infer_audio_model(audio_feature_csv, weight_file=r"E:\myworkspace\hxq_ade\weights\audio_1.h5")
