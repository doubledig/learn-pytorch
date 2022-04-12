import torch.nn
from torch.nn.functional import l1_loss


class HMLoss(torch.nn.Module):
    def __init__(self):
        super(HMLoss, self).__init__()

    def forward(self, o_hm, t_hm):
        pos_inds = t_hm.eq(1)
        neg_inds = ~pos_inds

        neg_weight = torch.pow(1 - t_hm, 4)
        num_pos = pos_inds.sum()
        neg_loss = torch.log(1 - o_hm) * torch.pow(o_hm, 2) * neg_weight * neg_inds
        neg_loss = neg_loss.sum()
        if num_pos == 0:
            return - neg_loss
        else:
            pos_loss = torch.log(o_hm) * torch.pow(1 - o_hm, 2) * pos_inds
            pos_loss = pos_loss.sum()
            return - (pos_loss + neg_loss) / num_pos


class RegL1Loss(torch.nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, o_wh, t_mask, t_ind, t_wh):
        s = t_mask.sum()
        if s == 0:
            return 0
        feat = o_wh.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        ind = t_ind.unsqueeze(2).expand(-1, -1, feat.size(2)).to(torch.int64)
        feat = feat.gather(1, ind)
        t_mask = t_mask.unsqueeze(2).expand_as(feat)
        loss = l1_loss(feat * t_mask, t_wh * t_mask, reduction='sum')
        loss = loss / s
        return loss
