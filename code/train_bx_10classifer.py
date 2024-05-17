import os
import math
import sys
import time, datetime
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

import torch.backends.cudnn as cudnn
import os.path as osp

# 导入自定义模块
import datasets
from utils_bx_10classifer import Logger
from model_v1_10classifer import efficientnet_b0 as create_model
from utils_bx_10classifer import train_one_epoch, evaluate

# 解析命令行参数
parser = argparse.ArgumentParser()
# 类别数量
parser.add_argument('--num_classes', type=int, default=10)
# 是否冻结部分层的权重
parser.add_argument('--freeze-layers', type=bool, default=False, help='是否冻结权重')
# 训练轮数
parser.add_argument('--epochs', type=int, default=100)
# 批大小
parser.add_argument('--batch-size', type=int, default=64)
# 学习率调整步长
parser.add_argument('--stepsize', type=int, default=20)
# 初始学习率
parser.add_argument('--lr', type=float, default=0.001)
# 最小学习率
parser.add_argument('--lrf', type=float, default=0.01)

# dataset
# 数据集名称
parser.add_argument('-d', '--dataset', type=str, default='birddataset', choices=['birddataset'])
# 数据加载器的工作线程数
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 0)")

# 预训练模型权重路径
# download model weights
# 链接: https://pan.baidu.com/s/1ouX0UmjCsmSx3ZrqXbowjw  密码: 090i
parser.add_argument('--weights', type=str, default='./weights_file/efficientnetb0-10.pth',
                    help='initial weights path')

# GPU设置
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
# misc
# 使用的 GPU 设备 ID
parser.add_argument('--gpu', type=str, default='0')
# 是否使用 CPU 运行
parser.add_argument('--seed', type=int, default=1)
# 随机种子
parser.add_argument('--use-cpu', action='store_true')
# 日志保存目录
parser.add_argument('--save-dir', type=str, default='log')
# 是否绘制特征图像
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")

args = parser.parse_args()


def main():

    # 创建文件夹
    os.makedirs("./weights", exist_ok=True)

    # 检查是否支持 GPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(comment='10classifer_1')
    # 权重
    # if os.path.exists("./weights") is False:
    #     os.makedirs("./weights")

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    print("use_gpu:", use_gpu)
    if args.use_cpu: use_gpu = False

    # class Unbuffered:
    #     def __init__(self, stream):
    #         self.stream = stream
    #
    #     def write(self, data):
    #         self.stream.write(data)
    #         self.stream.flush()
    # 将标准输出替换为不带缓冲的版本，这样tqdm就不会写入文件
    # sys.stdout = Unbuffered(sys.stdout)
    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    # 创建数据集对象
    dataset = datasets.create(
        name=args.dataset, batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers,
    )
    train_dataset, val_dataset = dataset.trainloader, dataset.testloader

    batch_size = args.batch_size
    # 设置数据加载器的工作线程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 创建模型
    model = create_model(num_classes=args.num_classes).to(device)
    # 如果存在预训练权重则载入
    # if args.weights != "":
    #     if os.path.exists(args.weights):
    #         weights_dict = torch.load(args.weights, map_location=device)
    #         # load_weights_dict = {k: v for k, v in weights_dict.items()
    #         #                      if model.state_dict()[k].numel() == v.numel()}
    #         load_weights_dict = {k: v for k, v in weights_dict.items()
    #                              if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    #         print(model.load_state_dict(load_weights_dict, strict=False))
    #     else:
    #         raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除最后一个卷积层和全连接层外，其他权重全部冻结
    #         if ("features.top" not in name) and ("classifier" not in name):
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    # 设置优化器
    optimizer = optim.SGD(pg, lr=0.01, momentum=0.9, weight_decay=1E-4)

    # cosine学习率调度器
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # 设置学习率调度器
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 初始化最好的准确率
    best_acc = 0.0

    start_time = time.time()
    for epoch in range(args.epochs):
        # 训练
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                train_loader=train_dataset,
                                                device=device,
                                                epoch=epoch)
        # 更新学习率
        scheduler.step()

        # 验证
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     val_loader=val_dataset,
                                     device=device,
                                     epoch=epoch,
                                     )
        # 将训练和验证指标写入TensorBoard
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 保存模型权重
        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch + 1))
        # 在每个 epoch 结束后，检查当前验证准确率是否高于历史最高准确率
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/best_model.pth")
            print(f"Saved best model with accuracy: {best_acc:.4f}")

    tb_writer.close()
    end_time = time.time()
    elapsed = round(end_time - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

# 程序入口
if __name__ == '__main__':
    main()
