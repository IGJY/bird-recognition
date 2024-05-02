import torch
# import torchvision
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset


class BIRDDATASET(object):

    def __init__(self, batch_size, use_gpu, num_workers):

        # 如果使用 GPU，则将数据加载到 GPU 内存中
        pin_memory = True if use_gpu else False

        # 加载特征和标签数据
        data_features = np.load(
            '../data/MFCC_train_combine.npy')
        data_labels = np.load('../data/MFCC_train_label_combine.npy')

        # 对标签进行 One-Hot 编码
        # my dataload one-hot encode
        data_labels = data_labels.reshape(-1, 1)
        dem = OneHotEncoder()
        dem.fit(data_labels)
        data_labels = dem.transform(data_labels).toarray()

        # data_labels = np.delete(data_labels, 0, axis=1)  # [[1,0,0,0,0]]
        # data_labels = np.argmax(data_labels, axis=1)

        # 划分数据集为训练集和测试集
        X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, test_size=0.3, random_state=10)

        # 将数据 reshape 成 PyTorch 所需的张量格式
        # mydata 60 63 paper 40 41
        X_train = X_train.reshape(X_train.shape[0], 3, 60, 63)
        X_test = X_test.reshape(X_test.shape[0], 3, 60, 63)

        # 创建训练集和测试集的 DataLoader 对象
        # trainloader =DataLoader(TensorDataset(torch.tensor(X_train).float(),torch.LongTensor(Y_train)), batch_size=batch_size, shuffle=False,
        #     num_workers=num_workers, pin_memory=pin_memory)
        trainloader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.LongTensor(Y_train)),
                                 batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                                 )
        testloader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.LongTensor(Y_test)),
                                batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory)

        # 保存 DataLoader 对象和数据类别数量
        self.trainloader = trainloader
        self.testloader = testloader
        # 这里的num_classes用于中心损失函数，应为10
        self.num_classes = 10

# 数据集工厂，用于创建不同的数据集对象
__factory = {
    'birddataset': BIRDDATASET,
}

# 创建数据集对象的函数
def create(name, batch_size, use_gpu, num_workers):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers)
