import math

from torch import nn
import torch


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, inputs, targets):
        pos_id = (targets == 1.0).float()
        neg_id = (targets == 0.0).float()
        pos_loss = pos_id * (inputs - targets).pow(2)
        neg_loss = neg_id * inputs.pow(2)
        pos_loss = pos_loss.sum(1).mean()
        neg_loss = neg_loss.sum(1).mean()
        return pos_loss, neg_loss


def generate_dxdywh(gt_label, w, h, s):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # compute the center, width and height
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1. or box_h < 1.:
        # print('A dirty data !!!')
        return False

    # map center point of box to the grid cell
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # compute the (x, y, w, h) for the corresponding grid cell
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = math.log(box_w)
    th = math.log(box_h)
    weight = 2.0 - (box_w / w) * (box_h / h)

    return grid_x, grid_y, tx, ty, tw, th, weight


def gt_creator(input_size, stride, label_lists=None):
    if label_lists is None:
        label_lists = []
    assert len(input_size) > 0 and len(label_lists) > 0
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    w = input_size[1]
    h = input_size[0]

    # We  make gt labels by anchor-free method and anchor-based method.
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = torch.zeros([batch_size, hs, ws, 1 + 1 + 4 + 1])

    # generate gt whose style is yolo-v1
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            gt_class = int(gt_label[-1])
            if gt_class == -1:
                continue
            result = generate_dxdywh(gt_label, w, h, s)
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight = result

                if grid_x < gt_tensor.shape[2] and grid_y < gt_tensor.shape[1]:
                    gt_tensor[batch_index, grid_y, grid_x, 0] = 1.0
                    gt_tensor[batch_index, grid_y, grid_x, 1] = gt_class
                    gt_tensor[batch_index, grid_y, grid_x, 2:6] = torch.tensor([tx, ty, tw, th])
                    gt_tensor[batch_index, grid_y, grid_x, 6] = weight

    gt_tensor = gt_tensor.reshape([batch_size, hs*ws, 7])

    return gt_tensor


def loss(pred_conf, pred_cls, pred_txtytwth, label):
    # 损失函数超参
    obj = 5.0
    noobj = 1.0
    # 创建损失函数
    conf_loss_function = MSELoss()
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    pred_conf = torch.sigmoid(pred_conf[:, :, 0])
    pred_cls = pred_cls.permute(0, 2, 1)
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]

    gt_obj = label[:, :, 0].float()
    gt_cls = label[:, :, 1].long()
    gt_txtytwth = label[:, :, 2:-1].float()
    gt_box_scale_weight = label[:, :, -1]
    # 计算前后景损失
    pos_loss, neg_loss = conf_loss_function(pred_conf, gt_obj)
    conf_loss = obj * pos_loss + noobj * neg_loss
    # 计算类别损失
    cls_loss = torch.mean(torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_obj, 1))
    # 预测框损失
    txty_loss = torch.mean(torch.sum(torch.sum(txty_loss_function(pred_txty, gt_txtytwth[:, :, :2]), 2) *
                                     gt_box_scale_weight * gt_obj, 1))
    twth_loss = torch.mean(torch.sum(torch.sum(twth_loss_function(pred_twth, gt_txtytwth[:, :, 2:]), 2) *
                                     gt_box_scale_weight * gt_obj, 1))
    txtytwth_loss = txty_loss + twth_loss
    # 总损失
    total_loss = conf_loss + cls_loss + txtytwth_loss
    return conf_loss, cls_loss, txtytwth_loss, total_loss
