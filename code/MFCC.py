import ntpath
import os
import sys
import time

import librosa
from librosa import feature
import numpy as np

import torch
from tqdm import tqdm


# 153516 138361 6904 8251

def get_MFCC(route):

    wav_file = route
    wav, sr = librosa.load(wav_file, sr=16000, duration=2.0)  # 具体的数据设置不需要修改，duration代表的就是提取的音频文件为2s

    # 计算音频信号的MFCC
    # label = float(ntpath.basename(os.path.dirname(route)))  # 数据的标签

    mfcc_delta0 = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=60)  # 提取MFCC特征，维度为60维
    mfcc_delta1 = librosa.feature.delta(mfcc_delta0)  # 一阶差分变换
    mfcc_delta2 = librosa.feature.delta(mfcc_delta1, order=2)  # 二阶差分变换，作用是为了增加动态信息，也是比较传统的做法

    mfcc = np.vstack([mfcc_delta0, mfcc_delta1, mfcc_delta2])  # 将MFCC和经过差分变换后的进行拼接,此时的shape为[-1,180,60,63]
    mfcc = mfcc.reshape([3, 60, 63])  # 修改shape 为[60，63]

    # return mfcc, label
    return mfcc
