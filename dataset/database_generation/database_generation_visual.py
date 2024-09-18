import os
from pathlib import Path

import numpy as np
import pandas as pd


def get_users_openface_feature_files(root_dir, is_patient=0):
    """
    获取目录下的所有openface生成对应的文件，文件名为人员ID
    :param root_dir: 目录
    :return:
    """

    root_path = Path(root_dir)
    csv_files = list(root_path.rglob("*.csv"))
    hog_files = list(root_path.rglob("*.hog"))

    users_feature_files = []
    for csv_file in csv_files:
        hog_file = csv_file.parent / f"{csv_file.stem }.hog"

        user_feature_files = {
            "user_id": csv_file.stem,
            "csv": csv_file,
            "hog": hog_file,
            "is_patient": is_patient,
        }

        users_feature_files.append(user_feature_files)
    print(f"users_feature_files: {users_feature_files}")
    return users_feature_files


def min_max_scaler(data):
    """recale the data, which is a 2D matrix, to 0-1"""
    return (data - data.min()) / (data.max() - data.min())


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def pre_check(data_df):
    data_df = data_df.apply(pd.to_numeric, errors="coerce")
    data_np = data_df.to_numpy()
    data_min = data_np[np.where(~(np.isnan(data_np[:, 5:])))].min()
    data_df.where(~(np.isnan(data_df)), data_min, inplace=True)
    return data_df


def load_all_feature(feature_path):
    all_feature_df = pre_check(pd.read_csv(feature_path, low_memory=False))
    return all_feature_df


def load_gaze(gaze_path):
    gaze_df = pre_check(pd.read_csv(gaze_path, low_memory=False))
    # process into format TxVxC
    gaze_coor = (
        gaze_df.iloc[:, 5:11].to_numpy().reshape(len(gaze_df), 2, 3)
    )  # 4 gaze vectors, 3 axes

    return gaze_coor


def load_keypoints(keypoints_path):
    fkps_df = pre_check(pd.read_csv(keypoints_path, low_memory=False))
    # process into format TxVxC
    x_coor = min_max_scaler(fkps_df[fkps_df.columns[301:369]].to_numpy())
    y_coor = min_max_scaler(fkps_df[fkps_df.columns[369:437]].to_numpy())
    z_coor = min_max_scaler(fkps_df[fkps_df.columns[437:505]].to_numpy())
    fkps_coor = np.stack([x_coor, y_coor, z_coor], axis=-1)

    return fkps_coor


def load_AUs(AUs_path):
    def check_AUs(data_df):
        data_df = data_df.apply(pd.to_numeric, errors="coerce")
        data_np = data_df.to_numpy()
        data_min = data_np[np.where(~(np.isnan(data_np[:, 505:539])))].min()
        data_df.where(~(np.isnan(data_df)), data_min, inplace=True)
        return data_df

    AUs_df = check_AUs(pd.read_csv(AUs_path, low_memory=False))
    AUs_features = min_max_scaler(AUs_df.iloc[:, 505:539].to_numpy())

    return AUs_features


def load_pose(pose_path):
    pose_df = pre_check(pd.read_csv(pose_path, low_memory=False))
    pose_coor = pose_df.iloc[:, 295:301].to_numpy()
    T, C = pose_coor.shape

    # initialize the final pose features which contains coordinate
    pose_features = np.zeros((T, C))
    # normalize the position coordinates part
    norm_part = min_max_scaler(pose_coor[:, :3])

    pose_features[:, :3] = norm_part  # normalized position coordinates
    pose_features[:, :3] = pose_coor[:, :3]  # head rotation coordinates
    pose_features = pose_features.reshape(T, 2, 3)  # 2 coordinates, 3 axes

    return pose_features


def get_num_frame(data, frame_size, hop_size):
    T = data.shape[0]
    if (T - frame_size) % hop_size == 0:
        num_frame = (T - frame_size) // hop_size + 1
    else:
        num_frame = (T - frame_size) // hop_size + 2
    return num_frame


def visual_padding(data, pad_size):
    if data.shape[0] != pad_size:
        size = tuple()
        size = size + (pad_size,) + data.shape[1:]
        padded_data = np.zeros(size)
        padded_data[: data.shape[0]] = data
    else:
        padded_data = data

    return padded_data


def save_user_feature(user_feature_dir, user_feature_dict):
    user_id = user_feature_dict["user_id"]
    csv = user_feature_dict["csv"]
    gazes = load_gaze(csv)
    key_points = load_keypoints(csv)
    aus = load_AUs(csv)
    pose = load_pose(csv)
    # hog = load_keypoints(csv)

    window_size = 60
    overlap_size = 10

    visual_sr = 30

    frame_size = window_size * visual_sr
    hop_size = (window_size - overlap_size) * visual_sr
    num_frame = get_num_frame(key_points, frame_size, hop_size)

    print("creating the data from the rest of the participants")
    # start sliding through and generating data
    for i in range(num_frame):
        frame_sample_fkps = visual_padding(
            key_points[i * hop_size : i * hop_size + frame_size], frame_size
        )
        frame_sample_gaze = visual_padding(
            gazes[i * hop_size : i * hop_size + frame_size], frame_size
        )
        frame_sample_AUs = visual_padding(
            aus[i * hop_size : i * hop_size + frame_size], frame_size
        )
        frame_sample_pose = visual_padding(
            pose[i * hop_size : i * hop_size + frame_size], frame_size
        )
        # frame_sample_hog = visual_padding(
        #     hog[i * hop_size: i * hop_size + frame_size], frame_size
        # )

        # start storing
        np.save(
            os.path.join(
                user_feature_dir, "facial_keypoints", f"{user_id}-{i:02}_kps.npy"
            ),
            frame_sample_fkps,
        )
        np.save(
            os.path.join(
                user_feature_dir, "gaze_vectors", f"{user_id}-{i:02}_gaze.npy"
            ),
            frame_sample_gaze,
        )
        np.save(
            os.path.join(user_feature_dir, "action_units", f"{user_id}-{i:02}_AUs.npy"),
            frame_sample_AUs,
        )
        np.save(
            os.path.join(
                user_feature_dir, "position_rotation", f"{user_id}-{i:02}_pose.npy"
            ),
            frame_sample_pose,
        )
        # np.save(
        #     os.path.join(user_feature_dir, "hog_features", f"{ID}-{i:02}_hog.npy"),
        #     frame_sample_hog,
        # )


def save_user_ids(dir_path, csv_path):
    user_ids = []
    gt_df = pd.read_csv(csv_path)
    for i in range(len(gt_df)):
        user_id = gt_df["user_id"][i]
        user_ids.append(user_id)
    np.save(
        os.path.join(dir_path, f"ids.npy"),
        user_ids,
    )


def save_user_labels(dir_path, csv_path):
    user_labels = []
    gt_df = pd.read_csv(csv_path)
    for i in range(len(gt_df)):
        user_label = gt_df["is_patient"][i]
        user_labels.append(user_label)
    np.save(
        os.path.join(dir_path, f"phq_binary_gt.npy"),
        user_labels,
    )


def save_user_base_info_to_csv(csv_path, data):
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    patient_openface_feature_root_dir = (
        r"E:\myworkspace\hxq_ade\data\hxq\video_subclip_patient"
    )

    normal_openface_feature_root_dir = (
        r"E:\myworkspace\hxq_ade\data\hxq\video_subclip_doctor"
    )

    patients_feature_files = get_users_openface_feature_files(
        patient_openface_feature_root_dir, is_patient=1
    )

    doctors_feature_files = get_users_openface_feature_files(
        normal_openface_feature_root_dir, is_patient=0
    )

    users_feature_files = []
    users_feature_files.extend(patients_feature_files)
    users_feature_files.extend(doctors_feature_files)

    # users_feature_files = users_feature_files[:10]

    user_ids = [users_dict["user_id"] for users_dict in users_feature_files]
    is_patient_labels = [users_dict["is_patient"] for users_dict in users_feature_files]
    user_base_info = {"user_id": user_ids, "is_patient": is_patient_labels}

    csv_path = r"E:\myworkspace\hxq_ade\dataset\clipped_data\user_info.csv"

    save_user_base_info_to_csv(csv_path, user_base_info)

    users_feature_dir = r"E:\myworkspace\hxq_ade\dataset\clipped_data"

    save_user_ids(users_feature_dir, csv_path)

    save_user_labels(users_feature_dir, csv_path)

    for user_feature_dict in users_feature_files:
        save_user_feature(users_feature_dir, user_feature_dict)
