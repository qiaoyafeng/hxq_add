import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(MLPBlock, self).__init__()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = nn.Linear(feature_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        output = self.softmax(self.layer4(x))
        return output


class Evaluator(nn.Module):
    def __init__(self, feature_dim, output_dim, predict_type):
        super(Evaluator, self).__init__()
        self.predict_type = predict_type
        self.evaluator = MLPBlock(feature_dim, output_dim)

    def forward(self, feats_avg):  # data: NCTHW
        probs = self.evaluator(feats_avg)  # Nxoutput_dim
        return probs


if __name__ == "__main__":
    ...
