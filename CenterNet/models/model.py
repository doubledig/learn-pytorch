import torch

from .backbone import resnet
from .model_dcn import PoseResNet

_backbone_factory = {
    'resnet18': resnet.resnet18,
    'resnet34': resnet.resnet34,
    'resnet50': resnet.resnet50,
    'resnet101': resnet.resnet101,
    'resnet152': resnet.resnet152
}


def create_model(backbone, task):
    if 'resnet' in backbone:
        if task == 'ctdet':
            # assert opt.dataset in ['coco']
            heads = {'hm': 80,
                     'wh': 2,
                     'reg': 2}
        else:
            heads = {}
        return PoseResNet(_backbone_factory[backbone](pretrained=True), heads, backbone)
    else:
        return None


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


def load_model(model_path, model, optimizer=None):
    checkpoint = torch.load(model_path)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'], model, optimizer
    return checkpoint['epoch'], model
