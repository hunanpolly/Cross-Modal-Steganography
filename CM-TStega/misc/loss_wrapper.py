import torch
import misc.utils as utils

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)

    def forward(self, fc_feats, att_feats, labels, masks, att_masks):
        out = {}
        loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:])
        out['loss'] = loss
        return out
