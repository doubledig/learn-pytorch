import torch
from torchvision import transforms

from CenterNet.utils.gaussian import gaussian_radius, draw_gaussian


class CTTransform:
    def __init__(self, split, p=0.5):
        super().__init__()
        self.p = p
        self.split = split
        self.input_h = 512
        self.input_w = 512
        self.output_h = self.input_h // 4
        self.output_w = self.input_w // 4
        self.valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]

    def __call__(self, img, target):
        num_classes = 80
        mean = (0.40789654, 0.44719302, 0.47026115)
        std = (0.28863828, 0.27408164, 0.27809835)
        # PLI图像Tensor化
        img = transforms.ToTensor()(img)
        # 处理图像
        channel, height, width = img.size()
        img = transforms.Resize([self.input_w, self.input_h], antialias=True)(img)
        # 数据增强
        flipped = False
        if self.split == 'train':
            if torch.rand(1) < self.p:
                flipped = True
                img = transforms.RandomHorizontalFlip(1.1)(img)
            if torch.rand(1) < self.p:
                img = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3)(img)
        img = transforms.Normalize(mean=mean, std=std)(img)

        # 标签制作
        hm = torch.zeros((num_classes, self.output_w, self.output_h))
        wh = torch.zeros((128, 2))
        ind = torch.zeros(128, dtype=torch.int)
        reg = torch.zeros((128, 2))
        reg_mask = torch.zeros(128, dtype=torch.uint8)

        k = 0
        for t in target:
            bbox = torch.tensor(t['bbox'])
            cls_id = self.valid_ids.index(t['category_id'])
            if flipped:
                bbox[0] = width - bbox[0] - bbox[2]
            bbox[0] = bbox[0] / width * self.output_w
            bbox[2] = bbox[2] / width * self.output_w
            bbox[1] = bbox[1] / height * self.output_h
            bbox[3] = bbox[3] / height * self.output_h
            radius = gaussian_radius(bbox[2], bbox[3])
            centerpoint = torch.tensor([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])
            centerpoint_int = centerpoint.to(torch.int)
            hm[cls_id] = draw_gaussian(hm[cls_id], centerpoint_int, radius)
            wh[k] = bbox[[2,3]]
            ind[k] = centerpoint_int[1] * self.output_w + centerpoint_int[0]
            reg[k] = centerpoint - centerpoint_int
            reg_mask[k] = 1
            k += 1

        ret = {'hm': hm,
               'reg': reg,
               'reg_mask': reg_mask,
               'ind': ind,
               'wh': wh}
        return img, ret
