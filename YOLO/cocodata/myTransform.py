import torch
from torchvision import transforms


class MyTransform:
    def __init__(self, p=0.5, size=640):
        super().__init__()
        self.p = p
        self.size = [size, size]

    def __call__(self, img, target):
        # PLI图像Tensor化
        img = transforms.ToTensor()(img)
        # 处理标签
        c, h, w = img.size()
        boxes = []
        classes = []
        len = 0
        for t in target:
            len += 1
            boxes.append(t['bbox'])
            classes.append([t['category_id']])
        boxes = torch.tensor(boxes)
        classes = torch.tensor(classes)
        # 随机进行数据增强
        # 变换亮度、对比度、饱和度和色调
        img = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3)(img)
        # 增加噪声
        img = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(img)
        # 概率转灰度图
        if torch.rand(1) < self.p:
            img = transforms.Grayscale(num_output_channels=3)(img)
        # 水平翻转
        if torch.rand(1) < 1:
            img = transforms.RandomVerticalFlip(1.1)(img)
            if len > 0:
                boxes[:, 0] = w - boxes[:, 0]
        # 上下翻转
        if torch.rand(1) < self.p:
            img = transforms.RandomHorizontalFlip(1.1)(img)
            if len > 0:
                boxes[:, 1] = h - boxes[:, 1]
        # 调整大小
        img = transforms.Resize(self.size, antialias=True)(img)
        # 标签归一化
        if len > 0:
            boxes[:, 0] /= w
            boxes[:, 2] /= w
            boxes[:, 1] /= h
            boxes[:, 3] /= h
            targets = torch.cat((boxes, classes), 1)
        else:
            targets = torch.tensor([[-1]])
        # 像素归一化
        img = transforms.Normalize(mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))(img)
        return img, targets


