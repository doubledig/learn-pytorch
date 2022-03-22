import argparse
import math
import os.path
import random
import time

import torch
import torchvision.datasets
from torch.utils.data.dataloader import DataLoader
from torch.backends import cudnn
from torch.nn.functional import interpolate

from cocodata.myTransform import MyTransform
from cocodata.collateFn import detection_collate
from models.MyYOLO import myyolo
from utils.loss import gt_creator


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO')
    parser.add_argument('--save_folder', default='weights/', type=str,
                        help='Gamma update for SGD')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--max_epoch', default=160, type=int,
                        help='training loop')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    return parser.parse_args()


def train():
    args = parse_args()
    # 训练参数储存地址
    path_to_save = os.path.join(args.save_folder, 'yolo')
    os.makedirs(path_to_save, exist_ok=True)
    # 设置cuda
    if args.cuda:
        if torch.cuda.is_available():
            print('use cuda')
            cudnn.benchmark = True
            device = torch.device('cuda')
        else:
            print('not support cuda, use cpu')
            device = torch.device('cpu')
    else:
        print('use cpu')
        device = torch.device('cpu')
    # 设置训练尺寸
    train_size = [640, 640]
    val_size = [416, 416]
    cfg = {'min_dim': [416, 416]}
    # 加载dataset和evaluator(计算AP)
    print('----------------------------------------------------------')
    print('Loading the dataset...')
    data_dir = '../data/coco'
    num_classes = 80
    dataset = torchvision.datasets.CocoDetection(root=data_dir + '/train2017',
                                                 annFile=data_dir + '/annotations/instances_train2017.json',
                                                 transforms=MyTransform(size=train_size[0]))
    # TODO evaluator
    print('Training model on: train2017')
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")
    # dataloader
    dataloader = DataLoader(dataset,
                            batch_size=32,
                            shuffle=True,
                            collate_fn=detection_collate,
                            num_workers=2)
    # 建立YOLO模型
    model = myyolo(device, input_size=train_size, num_classes=num_classes,
                   trainable=True)
    print('Let us train yolo on the coco dataset ......')
    model.to(device).train()
    # 读取训练数据继续训练
    if args.resume is not None:
        print('keep training model: %s' % args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=device))
    # 设置SGD
    base_lr = 1e-3
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=base_lr,
                                momentum=0.9,
                                weight_decay=5e-4)
    epoch_size = len(dataset) // 32
    # 记录开始训练时间
    t0 = time.time()
    # 设置学习轮数
    for epoch in range(args.max_epoch):
        # 更新学习率
        if epoch > 20:
            if epoch <= args.max_epoch - 20:
                lr = 0.00001 + 0.000495 * math.cos(math.pi * (epoch - 20) / (args.max_epoch - 20))
            else:
                lr = 0.00001
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # 读取数据进行训练
        for iter_i, (images, targets) in enumerate(dataloader):
            # 热身策略
            if epoch < 2:
                lr = base_lr * pow((iter_i + 1 + epoch * 32) / 64, 4)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            # 数据转换
            images = images.to(device)
            # 变换图像大小增强训练
            if iter_i % 10 == 0 and iter_i > 0:
                size = random.randint(10, 19) * 32
                train_size = [size, size]
                model.set_grid(train_size)
            images = interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            # 制作训练标签
            targets = [label.tolist() for label in targets]
            targets = gt_creator(input_size=train_size, stride=model.stride, label_lists=targets)
            targets = targets.to(device)
            # 回传损失
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, target=targets)
            # 反向传播
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # 输出记录
            if iter_i % 10 == 0:
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                      '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                      % (epoch + 1, args.max_epoch, iter_i, epoch_size, lr,
                         conf_loss.item(), cls_loss.item(), txtytwth_loss.item(), total_loss.item(), train_size[0],
                         t1 - t0),
                      flush=True)
                t0 = time.time()
            # TODO evaluator
            # 保存结果
            if (epoch + 1) % 10 == 0:
                print('Saving state, epoch:', epoch + 1)
                torch.save(model.state_dict(), os.path.join(path_to_save,
                                                            'YOLO_' + repr(epoch + 1) + '.pth'))
            pass
        pass


if __name__ == '__main__':
    train()
    pass
