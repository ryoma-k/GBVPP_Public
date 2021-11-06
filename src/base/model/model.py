from collections import OrderedDict
from copy import deepcopy
import math
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.ema = args['ema']['use']
        if self.ema:
            self.model = EMA(self.model, **args['ema']['ema_args'])
        self.bn_stats = True
        if args['freeze']:
            self.freeze()
            print('##### FREEZE TRAINING #####')
        self.diff_lr = args.get('diff_lr')
        setattr(self, '__getattr__', __getattr_over__)

    def __getattr_over__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                return getattr(self, name)
            except AttributeError:
                return getattr(self.model, name)

    def forward(self, *x, track_bn=True):
        self.set_bn_stats(track_bn)
        return self.model(*x)

    def update_ema(self):
        if hasattr(self, 'ema') and self.ema:
            self.model.update_ema()

    def set_bn_stats(self, track):
        if self.bn_stats != track:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.track_running_stats = track
        self.bn_stats = track

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if hasattr(m, 'weight'): m.weight.data.fill_(1)
            if hasattr(m, 'bias'): m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            if hasattr(m, 'weight'): nn.init.xavier_normal_(m.weight.data)
            if hasattr(m, 'bias'): m.bias.data.zero_()
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            for param in m.parameters():
                if len(param.shape) >= 2: nn.init.orthogonal_(param.data)
                else: nn.init.normal_(param.data)
        elif hasattr(m, 'parameters') and len([i for i in m.children()]) == 0:
            print(f'{m} passed')

    def get_parameters(self, lr=None):
        if self.diff_lr is not None:
            print(self.diff_lr)
            param_dicts = []
            default_parmas = []
            for n, p in self.named_parameters():
                default_scale_flag = True
                for pname in self.diff_lr:
                    if n.startswith((pname, f'model.{pname}')):
                        param_dict = {'params': p, **self.diff_lr[pname]}
                        param_dict['lr'] = param_dict.pop('scale') * lr
                        param_dicts.append(param_dict)
                        default_scale_flag = False
                if default_scale_flag:
                    default_parmas.append(p)
            param_dicts.append({'params': default_parmas})
            return [param_dicts]
        else:
            return [self.parameters()]

    def freeze(self):
        self.model.freeze()

    def unreeze(self):
        self.model.unfreeze()


class EMA(nn.Module):
    def __init__(self, model, decay):
        super().__init__()
        self._model = model
        self.decay = decay
        self._shadow = deepcopy(self._model)
        for param in self._shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update_ema(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self._model.named_parameters())
        shadow_params = OrderedDict(self._shadow.named_parameters())
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self._model.named_buffers())
        shadow_buffers = OrderedDict(self._shadow.named_buffers())
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, _buffer in model_buffers.items():
            shadow_buffers[name].copy_(_buffer)

    def forward(self, *x, **args):
        if self.training:
            return self._model(*x, **args)
        else:
            return self._shadow(*x, **args)


class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def load_state_dict(self, state_dict):
        try:
            super().load_state_dict(state_dict)
        except:
            self.module.load_state_dict(state_dict)

    def state_dict(self):
        return self.module.state_dict()
