import os
import torch
from torch.backends import cudnn
from torch.utils.data.dataloader import DataLoader

from utils.opts import opts
from datasets.dataset_factory import get_dataset
from models.model import create_model, save_model, load_model
from trains.trainer_factory import get_trainer


def train(opt):
    # 设置随机种子
    torch.manual_seed(opt.seed)
    path_to_save = os.path.join(opt.save_folder, opt.dataset+'/', opt.backbone+'/')
    os.makedirs(path_to_save, exist_ok=True)
    # gpu
    if opt.gpus[0] >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        opt.device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        opt.device = torch.device('cpu')
    # dataset
    train_dataset = get_dataset(opt.dataset, opt.task, 'train')

    # 建立训练模型
    print('Creating model...')
    model = create_model(opt.backbone, opt.task)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    if opt.load_model != '':
        start_epoch, model, optimizer = load_model(opt.load_model,
                                                   model,
                                                   optimizer)

    trainer = get_trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.device)

    # dataloader
    print('Setting up data...')
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    lr_step = [90,120]
    for epoch in range(start_epoch + 1, 141):
        trainer.train(epoch, train_loader)
        save_model(os.path.join(path_to_save, 'model_last.pth'),
                   epoch, model, optimizer)
        if epoch in lr_step:
            save_model(os.path.join(path_to_save, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1**(lr_step.index(epoch)+1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


if __name__ == '__main__':
    opt = opts().parse()

    train(opt)
