from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torchvision import transforms


# 检查CUDA是否可用
from common.constants import ALL_LABELS_DICT, ALL_LABELS_DESC_DICT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_loss = 1
current_loss = 1


# 定义CNN模型
class MultiClassNet(nn.Module):
    def __init__(self):
        super(MultiClassNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 6)  # 6类分类

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


def padding(data, pad_size=120):
    if data.shape[0] < pad_size:
        size = tuple()
        size = size + (pad_size,) + data.shape[1:]
        padded_data = np.zeros(size)
        padded_data[: data.shape[0]] = data
    else:
        padded_data = data[:pad_size]

    return padded_data


# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths  # 存储每个样本文件路径
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 读取CSV文件中的特征
        item_file = self.file_paths[idx]
        print(f"item_file: {item_file}")
        data = pd.read_csv(item_file)
        data = padding(data.iloc[:, 2:30], pad_size=28)
        print(f"data shape: {data.shape}")
        features = np.array(data.values.astype(np.float32))
        label = torch.tensor(self.labels[idx])
        features = torch.tensor(features).view(1, 28, 28)
        return features, label


# 保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)


# 加载模型
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()


# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    global best_loss
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        epoch_loss = loss.item()
        if loss.item() < best_loss:
            save_model(model, f"cnn_model_{epoch_loss}.pth")
            best_loss = epoch_loss


def get_data_file_label_list():
    extension = ".csv"
    all_labels = ["normal", "F20", "F31", "F32", "F41", "F42"]
    all_labels_dict = {"normal": 0, "F20": 1, "F31": 2, "F32": 3, "F41": 4, "F42": 5}
    file_list = []
    label_list = []
    data_file_dir = r"E:\myworkspace\hxq_ade\data\hxq\multi_class"
    for label in all_labels:
        label_file_dir = Path(data_file_dir) / label
        label_files = list(label_file_dir.rglob("*" + extension))
        labels = [all_labels_dict[label] for i in range(len(label_files))]
        file_list.extend(label_files)
        label_list.extend(labels)

    return file_list, label_list


# 推理
def infer(model, input_data):
    with torch.no_grad():
        output = model(input_data)
        _, predicted = torch.max(output, 1)
        print(f"infer: output: {output}, {_, predicted}")
        return predicted


if __name__ == "__main__":
    # 参数设置
    file_paths, labels = get_data_file_label_list()

    print(f"file_paths: {file_paths}, labels: {labels}")

    batch_size = 50
    num_epochs = 50
    learning_rate = 0.001
    model_save_path = "cnn_model.pth"

    # 数据转换
    transform = transforms.Compose(
        [
            transforms.Resize((120, 535)),
            transforms.ToTensor(),
        ]
    )

    # 数据加载与模型训练
    dataset = CustomDataset(file_paths, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MultiClassNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    # train_model(model, dataloader, criterion, optimizer, num_epochs)

    # 保存训练好的模型
    # save_model(model, model_save_path)

    # 加载模型进行推理
    loaded_model = MultiClassNet()
    load_model(loaded_model, model_save_path)

    # 示例推理 all_labels_dict = {"normal": 0, "F20": 1, "F31": 2, "F32": 3, "F41": 4, "F42": 5}
    normal_csv = r"E:\myworkspace\hxq_ade\data\hxq\multi_class\normal\_SoX60mKmr9joMfcOT2El8_EC71BFD2-D846-49B6-BA09-137B40DC09A2_1709041923756_doctor.csv"
    f20_csv = r"E:\myworkspace\hxq_ade\data\hxq\multi_class\F20\9o5atvJY0S7nZ4KMCYAlTA_CB4A886E-FF60-4C52-AEA2-48C9AD504104_1681565414504.csv"
    f31_csv = r"E:\myworkspace\hxq_ade\data\hxq\multi_class\F31\2NlXXKkSTBArMcAcgN8I87_f215c9ca-8404-4b54-b405-ae73f2e4f4d6_1694005229535.csv"
    f32_csv = r"E:\myworkspace\hxq_ade\data\hxq\multi_class\F32\_SoX60mKmr9joMfcOT2El8_EC71BFD2-D846-49B6-BA09-137B40DC09A2_1709041923756.csv"
    f41_csv = r"E:\myworkspace\hxq_ade\data\hxq\multi_class\F41\KzdMBYL6SeAmEBauFukGi7_1777dd18-a820-42ba-89e6-51ef8c52283a_1685354605923.csv"
    f42_csv = r"E:\myworkspace\hxq_ade\data\hxq\multi_class\F42\1xdBelP3v7AlcgqS2-4jB7_2c3b488a-fe81-428f-8b9d-bf9db3ebc57e_1686314511083.csv"
    test_csv = f32_csv
    data = pd.read_csv(test_csv)
    data = padding(data.iloc[:, 2:30], pad_size=28)
    print(f"data shape: {data.shape}")
    features = np.array(data.values.astype(np.float32))
    print(f"features: {features.shape}")
    features = torch.tensor(features).view(1, 28, 28).unsqueeze(0)
    prediction = infer(loaded_model, features)
    print(f"prediction: {prediction}")

    inverted_labels_dict = {value: key for key, value in ALL_LABELS_DICT.items()}
    label_id = int(prediction)
    state = inverted_labels_dict[label_id]
    description = ALL_LABELS_DESC_DICT[state]
    res = {
        "index": label_id,
        "state": state,
        "description": description,
    }
    print(f"res: {res}")
