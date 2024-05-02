# your_model_loading_script.py

import torch
from model_v1_10classifer import efficientnet_b0 as create_model


def load_model_and_weights(weights_path):
    # 创建模型实例
    model = create_model(num_classes=10)  # 这里的参数可能需要根据您的模型结构进行调整

    # 加载权重
    if torch.cuda.is_available():
        # 如果有 GPU，则加载到 GPU 上
        model.load_state_dict(torch.load(weights_path))
    else:
        # 如果没有 GPU，则加载到 CPU 上
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    # 返回加载好权重的模型实例
    return model
