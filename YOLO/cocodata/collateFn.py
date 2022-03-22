import torch


def detection_collate(batch):
    # 对每批数据进行拼接
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs, 0), targets
