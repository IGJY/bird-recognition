import os
import sys
import errno
import shutil
import os.path as osp

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))


class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def train_one_epoch(model, optimizer, train_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()
    sample_num = 0
    train_bar = tqdm(train_loader, file=sys.stdout)

    for step, data in enumerate(train_bar):
        features, labels = data

        sample_num += labels.size(0)
        pred = model(features.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        # my dataload
        label = torch.max(labels, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum().item()

        loss = loss_function(pred, labels.float().to(device))

        optimizer.zero_grad()
        loss.backward()
        accu_loss += loss.detach()

        train_bar.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch + 1,
                                                                             accu_loss.item() / (step + 1),
                                                                             accu_num.item() / sample_num * 100)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, val_loader, device, epoch):
    model.eval()
    acc_loss = AverageMeter()
    loss_function = torch.nn.CrossEntropyLoss()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    sample_num = 0

    val_bar = tqdm(val_loader, file=sys.stdout)

    for step, data in enumerate(val_bar):
        features_val, labels = data
        sample_num += labels.size(0)

        pred = model(features_val.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        # my dataload
        label = torch.max(labels, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum()

        loss = loss_function(pred, labels.float().to(device))

        acc_loss.update(loss.item(), labels.size(0))
        accu_loss += loss

        val_bar.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch + 1,
                                                                           accu_loss.item() / (step + 1),
                                                                           accu_num.item() / sample_num * 100)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
