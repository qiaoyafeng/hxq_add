import os
import time

import numpy as np
import pandas as pd

import audb
import audiofile
import opensmile

file = "audio.wav"
signal, sampling_rate = audiofile.read(file)
print(f"signal: {signal}, sampling_rate: {sampling_rate}")
re_resample_rate = 8000

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    sampling_rate=re_resample_rate,
    resample=True,
)

feature_names = smile.feature_names
print(f"feature_names: {feature_names}, feature_names len: {len(feature_names)}")

process_signal = smile.process_signal(
    signal,
    sampling_rate,
)

print(f"process_signal: {process_signal}")
