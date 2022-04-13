import os
import torch
import torchvision

from CenterNet.datasets.coco import CocoDetection
from CenterNet.detectors.detector_factory import detector_factory
from CenterNet.models.model import create_model, load_model
from CenterNet.utils.opts import opts


def test(opt):
    if opt.gpus[0] >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    pass
    # dataset 自己根据需求修改
    val_dataset = CocoDetection(
        root='../data/coco/val2017',
        annFile='../data/coco/annotations/instances_val2017.json'
    )
    # model
    model = create_model(opt.backbone, opt.task)
    if opt.load_model == '':
        print('no model load')
        return
    _, model = load_model(opt.load_model, model)
    model.to(opt.device)
    model.eval()

    detector = detector_factory[opt.task](opt)
    results, avg_time_stats = detector.run(val_dataset, model)
    print('tot_time: {:.3f}'.format(avg_time_stats['tot'].sum))
    for t in avg_time_stats:
        print('{}_avg_time: {:.3f}'.format(t, avg_time_stats[t].avg))
    val_dataset.run_eval(results, 'json/')


if __name__ == '__main__':
    # 需要确认backbone和加载的文件相同
    opt = opts().parse()
    test(opt)
