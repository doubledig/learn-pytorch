import time
from typing import Any, Tuple

import torch
from torchvision import transforms

from CenterNet.models.decode import ctdet_decode
from CenterNet.utils.AverageMeter import AverageMeter


class CtdetDetector:
    def __init__(self, opt):
        self.device = opt.device
        if opt.dataset == 'coco':
            self.valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                              14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                              24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                              37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                              48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                              58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                              72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                              82, 84, 85, 86, 87, 88, 89, 90]

    def run(self, dataset, model) -> Tuple[Any, Any]:
        mean = (0.40789654, 0.44719302, 0.47026115)
        std = (0.28863828, 0.27408164, 0.27809835)
        time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post']
        avg_time_stats = {t: AverageMeter() for t in time_stats}
        results = {}
        for i in range(len(dataset)):
            a_start_time = time.time()
            img_id, img = dataset.load_image(i)
            load_time = time.time() - a_start_time
            start_time = time.time()
            img = transforms.ToTensor()(img)
            img = img.to(self.device)
            c, h, w = img.size()
            img = transforms.Resize([512, 512], antialias=True)(img)
            img = transforms.Normalize(mean=mean, std=std)(img)
            img.unsqueeze_(0)
            torch.cuda.synchronize()
            pre_time = time.time() - start_time
            start_time = time.time()
            with torch.no_grad():
                output = model(img)
            output['hm'].sigmoid_()
            torch.cuda.synchronize()
            net_time = time.time() - start_time
            start_time = time.time()
            dets = ctdet_decode(output['hm'], output['wh'], output['reg'], K=100)
            dec_time = time.time() - start_time
            start_time = time.time()
            result = {}
            dets = dets.view(-1, dets.size(2))
            dets[:, 0:4:2] = dets[:, 0:4:2] / 512 * w
            dets[:, 1:5:2] = dets[:, 1:5:2] / 512 * h
            for ind in range(80):
                cla = dets[:, -1] == ind
                if cla.sum() > 0:
                    result[self.valid_ids[ind]] = dets[cla, 0:5]
            post_time = time.time() - start_time
            results[img_id] = result
            tot_time = time.time() - a_start_time
            ret = {'tot': tot_time, 'load': load_time,
                   'pre': pre_time, 'net': net_time, 'dec': dec_time,
                   'post': post_time}
            for t in avg_time_stats:
                avg_time_stats[t].update(ret[t])
        return results, avg_time_stats
