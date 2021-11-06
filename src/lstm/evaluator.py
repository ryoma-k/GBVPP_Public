import torch
import torch.nn as nn
from base.trainers import Loss, Criterion


class GBVPPEvaluator(Loss, Criterion):
    def __init__(self, lambdas, summary):
        Loss.__init__(self, lambdas, summary)
        self.L1 = nn.L1Loss(reduction='none')
        RCs = torch.tensor([[i,j] for i in range(3) for j in range(3)])
        self.register_buffer('RCs', RCs)
        self.RC2name = [f'R{R} C{C}' for R in [5,20,50] for C in [10,20,50]]
        self.best_score = 1e9

    def masked_l1(self, outputs, labels, **kargs):
        mask = labels['mask']
        label = labels['label']
        output = outputs[...,0]
        loss = self.L1(output[mask], label[mask])
        return loss.mean()

    def masked_laplace(self, outputs, labels, **kargs):
        mask = labels['mask']
        label = labels['label']
        mu = outputs[...,0][mask]
        b = outputs[...,1][mask]
        p = 1 / (2*b) * (- (label[mask] - mu).abs() / b).exp()
        loss = - torch.log(p + 1e-12)
        return loss.mean()

    def criterion(self, outputs, labels, stage):
        values = {}
        mask = labels['u_out'] == 0
        label = labels['label']
        outputs_mu = outputs[...,0]
        values['l1'] = self.L1(outputs_mu[mask], label[mask]).mean()
        values['l1_all'] = self.L1(outputs_mu, label).mean()
        values['b_mean'] = outputs[...,1].mean()
        diffs = (outputs_mu - label).abs()
        diffs[~mask] = 0
        diffs = diffs.sum(-1) / mask.sum(-1)
        for name, RC in zip(self.RC2name, self.RCs):
            RC_mean = diffs[(labels['cate_v'] == RC).all(-1)[:,0]].mean()
            if not torch.isnan(RC_mean):
                values[name] = RC_mean
        for key in values:
            self.log(values[key], key, stage)

    def is_best_score(self):
        current_score = self.summary.last_mean_vs.get('val/l1', 1e9)
        if self.best_score > current_score:
            self.best_score = current_score
            return True
        else:
            return False
