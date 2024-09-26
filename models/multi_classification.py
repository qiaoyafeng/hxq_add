from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torchvision import transforms


# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
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


# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


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


# 参数设置
file_paths, labels = get_data_file_label_list()

print(f"file_paths: {file_paths}, labels: {labels}")


batch_size = 5
num_epochs = 10
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

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, dataloader, criterion, optimizer, num_epochs)


# 保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)


# 加载模型
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()


# 保存训练好的模型
save_model(model, model_save_path)

# 加载模型进行推理
loaded_model = SimpleCNN()
load_model(loaded_model, model_save_path)


# 推理示例
def infer(model, input_data):
    with torch.no_grad():
        output = model(input_data)
        _, predicted = torch.max(output, 1)
        print(f"infer: output: {output}, {_, predicted}")
        return predicted


# 示例推理
test_csv = "E:\myworkspace\hxq_ade\data\hxq\multi_class\F32\_UCetHaSH29GQYWVB-Mk48_c8f620d5-0835-44a2-875e-751d8f65d24b_1705392686557.csv"
data = pd.read_csv(test_csv)
data = padding(data.iloc[:, 2:30], pad_size=28)
print(f"data shape: {data.shape}")
features = np.array(data.values.astype(np.float32))
print(f"features: {features.shape}")
features = torch.tensor(features).view(1, 28, 28).unsqueeze(0)
prediction = infer(loaded_model, features)
print(f"prediction: {prediction}")
