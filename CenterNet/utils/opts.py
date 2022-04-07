import argparse
import os

'''
    进行运行时相关参数的设置
'''


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # 基础环境参数
        self.parser.add_argument('--task', default='ctdet',
                                 help='ctdet')
        self.parser.add_argument('--save_folder', default='./weights/', type=str,
                                 help='save_folder')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')

        # 'ctdet'目标检测
        self.parser.add_argument('--dataset', default='coco',
                                 help='coco')

        # 系统参数
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=2,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed', type=int, default=100,
                                 help='random seed')

        # 模型
        self.parser.add_argument('--backbone', default='resnet18',
                                 help='resnet18 | resnet34 | resnet50 | resnet101 |'
                                      'resnet152')
        # 训练
        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='batch size')



    def parse(self):
        opt = self.parser.parse_args()
        # 设置GPU参数
        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]

        return opt
