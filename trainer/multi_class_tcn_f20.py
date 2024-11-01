import os

from tensorflow.keras.callbacks import Callback
import pandas as pd
import random
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU, Conv1D, Dense
from tcn import TCN, tcn_full_summary
from sklearn.metrics import mean_squared_error  # 均方误差
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers

matplotlib.use("TkAgg")


my_seed = 666
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)


tran_data = []
hdr_path = r"E:\myworkspace\hxq_ade\data\hxq\multi_class_tcn\F20_hdr"
for file in os.listdir(hdr_path):
    label = int(file.split("_")[0])
    feature_path = hdr_path + "/" + file
    df = pd.read_csv(feature_path)
    print(f"file: {file}-------label: {label}")
    labels = pd.DataFrame({"label": [label] * df.shape[0]})
    df2 = pd.concat([df.iloc[:, 1:], labels], axis=1)
    tran_data.append(df2)


df = pd.concat(tran_data, ignore_index=True)

df = df.sample(frac=1.0).reset_index(drop=True)


len1 = int(df.shape[0] * 0.9)
train = df[:len1]
test = df[len1:]

# 不对y进行归一化
X_train = train.iloc[:, :-1]
Y_train = train.iloc[:, -1:]
X_test = test.iloc[:, :-1]
Y_test = test.iloc[:, -1:]


# #归一化
df = df.iloc[:, :-1]
scaler = MinMaxScaler(feature_range=(0, 1)).fit(df)
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

Y_test = Y_test.reset_index(drop=True)
Y_test.head(10)


x_train = X_train.values.reshape([X_train.shape[0], 1, X_train.shape[1]])
y_train = Y_train.values
x_test = X_test.values.reshape([X_test.shape[0], 1, X_test.shape[1]])
y_test = Y_test.values

print(f"train and test shape: {x_train.shape,y_train.shape,x_test.shape,y_test.shape}")


def rmse(y_pred, y_true):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

tcn_name = "tcn_f20"
batch_size = None
timesteps = x_train.shape[1]
timesteps = 10
input_dim = x_train.shape[2]  # 输入维数
tcn = Sequential()
input_layer = Input(batch_shape=(batch_size, timesteps, input_dim))
tcn.add(input_layer)
tcn.add(
    TCN(
        nb_filters=32,  # 在卷积层中使用的过滤器数。可以是列表。
        kernel_size=3,  # 在每个卷积层中使用的内核大小。
        nb_stacks=1,  # 要使用的残差块的堆栈数。
        dilations=[2**i for i in range(4)],  # 扩张列表。示例为：[1、2、4、8、16、32、64]。
        # 用于卷积层中的填充,值为'causal' 或'same'。
        # “causal”将产生因果（膨胀的）卷积，即output[t]不依赖于input[t+1：]。当对不能违反时间顺序的时序信号建模时有用。
        # “same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
        padding="causal",
        use_skip_connections=True,  # 是否要添加从输入到每个残差块的跳过连接。
        dropout_rate=0.3,  # 在0到1之间浮动。要下降的输入单位的分数。
        return_sequences=False,  # 是返回输出序列中的最后一个输出还是完整序列。
        activation="relu",  # 残差块中使用的激活函数 o = Activation(x + F(x)).
        kernel_initializer="he_normal",  # 内核权重矩阵（Conv1D）的初始化程序。
        use_batch_norm=True,  # 是否在残差层中使用批处理规范化。
        #         use_layer_norm=False, #是否在残差层中使用层归一化。
        name=tcn_name,  # 使用多个TCN时，要使用唯一的名称
    )
)

# sgd = optimizers.SGD(learning_rate=0.01, clipvalue=0.5)
sgd = optimizers.SGD()

tcn.add(tf.keras.layers.Dense(1))
# tcn.compile(optimizer=sgd, loss='mse', metrics=['mae'])
tcn.compile(optimizer='sgd', loss='mse', metrics=['mae'])
tcn.summary()


early_stopping = EarlyStopping(monitor="val_mae", patience=200, verbose=2)
checkpoint = ModelCheckpoint(
    f"{tcn_name}.keras",
    monitor="val_mae",
    verbose=1,
    save_best_only=True,
    mode="min",
)
callbacks_list = [checkpoint, early_stopping]

history = tcn.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=200,
    batch_size=128,
    callbacks=callbacks_list,
)

tcn.summary()


history_keys = history.history.keys()
print(f"history_keys: {history_keys}")


history.history.keys()  # 查看history中存储了哪些参数
plt.plot(
    history.epoch, history.history.get("val_mae"), "r", label="val_mae"
)  # 画出随着epoch增大loss的变化图

plt.show()

plt.plot(
    history.epoch, history.history.get("mae"), "g", label="mae"
)  # 画出随着epoch增大准确率的变化图
plt.legend(loc=0, ncol=2)
plt.show()


score_test = tcn.evaluate(x_test, y_test)
print(f"score_test: {score_test}")


if __name__ == "__main__":
    pass
