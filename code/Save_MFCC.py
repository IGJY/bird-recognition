# 保存GFCC到本地
import os
import sys
import librosa
import numpy as np
from tqdm import tqdm
import torch
from MFCC_S import get_MFCC

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def save_MFCC():

    files = '..\\data\\BirdsSong-10spec-graduation'  # 音频文件的地址，为各个分类的上一级
    all_mfcc = []
    all_label = []
    # jishu = 0
    for file in os.listdir(files):
        wav_files = files + '\\' + file
        print(wav_files)
        for wav_file in tqdm(os.listdir(wav_files)):
            file_1 = wav_files + '\\' + wav_file
            # print(ntpath.basename(os.path.dirname(file_1)))
            mfcc, label = get_MFCC(route=file_1)
            # print(gfcc)
            all_mfcc.append(mfcc)
            all_label.append(label)
            # print(all_label)
            # jishu +=1
            # print(jishu)
    mfcc_train = np.array(all_mfcc)
    label_train = np.array(all_label)
    mfcc_train = torch.tensor(mfcc_train)
    label_train = torch.tensor(label_train)
    label_train = torch.squeeze(label_train)

    np.save('../data/MFCC_train/MFCC_train_sample.npy', mfcc_train)
    np.save("../data/MFCC_train/MFCC_train_label.npy", label_train)

    print("保存完毕")
