import time

import torch


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, img, target):
        outputs = self.model(img)
        loss, loss_stats = self.loss(outputs, target)
        return loss, loss_stats


class BaseTrainer(object):
    def __init__(self, opt, model, optimizer):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self.get_losses()
        self.model_with_loss = ModelWithLoss(model, self.loss)

    def get_losses(self):
        raise NotImplementedError

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            self.model_with_loss = torch.nn.DataParallel(
                self.model_with_loss, device_ids=gpus).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device)

    def train(self, epoch, data_loader):
        return self.run_epoch(epoch, data_loader)

    def run_epoch(self, epoch, data_loader):
        model_with_loss = self.model_with_loss
        model_with_loss.train()

        num_iters = len(data_loader)
        t0 = time.time()
        for iter_id, (images, targets) in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            images = images.to(self.opt.device)
            for k in targets:
                targets[k] = targets[k].to(self.opt.device)
            loss, loss_stats = model_with_loss(images, targets)
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if iter_id % 20 == 0:
                t1 = time.time()
                print('[Epoch %d][Iter %d/%d]'
                      '[Loss: hm_loss %.2f || wh_loss %.2f || off_loss %.2f || total %.2f || time: %.2f]'
                      % (epoch, iter_id, num_iters,
                         loss_stats['hm_loss'].mean(), loss_stats['wh_loss'].mean(), loss_stats['off_loss'].mean(),
                         loss, t1 - t0),
                      flush=True)
                t0 = time.time()
            del loss, loss_stats

