import torch.nn

from .base_trainer import BaseTrainer
from models.losses import HMLoss, RegL1Loss


class CtdetLoss(torch.nn.Module):
    def __init__(self, device):
        super(CtdetLoss, self).__init__()
        self.device = device
        self.crit = HMLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = self.crit_reg

    def forward(self, outputs, targets):
        outputs = outputs[0]
        hm_loss, wh_loss, off_loss = 0, 0, 0
        # 激活归一热点图
        outputs['hm'] = self.hm_sigmoid(outputs['hm'])
        hm_loss += self.crit(outputs['hm'], targets['hm'])
        # wh_loss
        wh_loss += self.crit_reg(outputs['wh'], targets['reg_mask'],
                                 targets['ind'], targets['wh'])

        off_loss += self.crit_reg(outputs['reg'], targets['reg_mask'],
                                  targets['ind'], targets['reg'])

        loss = 1 * hm_loss + 0.1 * wh_loss + 1 * off_loss
        loss_stats = {'hm_loss': hm_loss,
                      'wh_loss': wh_loss,
                      'off_loss': off_loss}
        return loss, loss_stats

    def hm_sigmoid(self, x):
        x.sigmoid_()
        # x = torch.where(x < 1e-4, torch.zeros(x.size(), device=self.device), x)
        x = torch.where(x > 1 - 1e-4, torch.ones(x.size(), device=self.device), x)
        return x


class CtdetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer):
        super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def get_losses(self):
        loss_states = ['hm_loss', 'wh_loss', 'off_loss']
        loss = CtdetLoss(self.opt.device)
        return loss_states, loss
