import torch
from torch.nn import functional


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat)
    return heat * keep


def _topk(heat, K):
    batch, cat, height, width = heat.size()

    topk_scores, topk_inds = torch.topk(heat.view(batch, -1), K)
    topk_clses = torch.div(topk_inds, height * width, rounding_mode='floor')
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode='floor')
    topk_xs = topk_inds % width

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


# 80 128 128  2 128 128
def ctdet_decode(heat, wh, reg, K=128):
    batch, cat, height, width = heat.size()
    # nms
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    inds = inds.unsqueeze(2).expand(batch, K, 2)
    # reg
    reg = reg.permute(0, 2, 3, 1).contiguous()
    reg = reg.view(batch, -1, 2)
    reg = reg.gather(1, inds)
    # y,x
    xs = xs.view(batch, K, 1) + reg[:,:, 0:1]
    ys = ys.view(batch, K, 1) + reg[:,:, 1:2]
    # wh
    wh = wh.permute(0, 2, 3, 1).contiguous()
    wh = wh.view(batch, -1, 2)
    wh = wh.gather(1, inds)

    clses = clses.view(batch, K, 1)
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        wh[..., 0:1],
                        wh[..., 1:2]], dim=2)
    return torch.cat([bboxes, scores, clses], dim=2)
