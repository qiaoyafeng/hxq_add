from models.convlstm import ConvLSTMAudio
from models.evaluator import Evaluator
import torch.nn as nn

audio_net = ConvLSTMAudio(
    input_dim=80,
    output_dim=256,
    conv_hidden=256,
    lstm_hidden=256,
    num_layers=4,
    activation="relu",
    norm="bn",
    dropout=0.5,
)

evaluator = Evaluator(feature_dim=256, output_dim=25, predict_type="phq-score")


print(f"audio_net: {audio_net}")

print(f"evaluator: {evaluator}")
