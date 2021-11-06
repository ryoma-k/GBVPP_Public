import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, lambdas, summary=None):
        super().__init__()
        for k, v in lambdas.items():
            setattr(self, k, v)
        self.summary = summary

    def __call__(self, *inputs, mode, stage, lam_mode='', note='', write=True, **args):
        try:
            if len(lam_mode): lam_mode = '_' + lam_mode
            _lambda = eval('self.lambda_' + mode + lam_mode)
            func = eval(f'self.{mode}')
        except AttributeError:
            print(f'mode : {mode} does not exist.')
            raise Exception
        loss = func(*inputs, mode=mode, stage=stage, **args)
        if loss.numel() == 1 and (torch.isnan(loss) or loss == float('inf')):
            import pdb; pdb.set_trace()
        loss = loss * _lambda
        note += '_'
        if write and self.summary is not None:
            self.summary.add_scalar(f'{stage}/loss_{mode}{lam_mode}{note}', loss.item())
        return loss

    def log(self, value, mode, stage, note=''):
        if self.summary is not None:
            self.summary.add_scalar(f'{stage}/{mode}{note}', value.item())
