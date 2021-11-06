from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Summary():
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
        self.counter_ep = {}
        self.counter_tot = {}
        self.value_ep = {}
        self.last_mean_vs = {}
        self.epoch = 0

    def add_scalar(self, name, value, num=1, take_average=True):
        '''
        stores scalar value and takes average.
        name: type of scalar, e.g.) "loss"
        value: scalar
        num: if the value has already taken averaged, set this valuable to its original size.
        '''
        if take_average:
            self.counter_ep[name] = self.counter_ep.get(name, 0) + num
        self.counter_tot[name] = self.counter_tot.get(name, 0) + 1
        self.value_ep[name] = self.value_ep.get(name, 0.) + value * num
        if self.writer is not None:
            self.writer.add_scalar(name, value, self.counter_tot[name])

    def add_text(self, *args):
        if self.writer is not None:
            self.writer.add_text(*args)

    def add_image(self, *args):
        if self.writer is not None:
            self.writer.add_image(*args)

    def add_histogram(self, *args):
        if self.writer is not None:
            self.writer.add_histogram(*args)

    def end_epoch(self, epoch):
        '''
        call this method at the end of the epoch.
        '''
        mean_vs = {}
        for key in self.counter_ep:
            v = self.value_ep[key]
            c = self.counter_ep[key]
            if self.writer is not None:
                self.writer.add_scalar('ep/' + key, v/max(c, 1e-6), epoch)
            mean_vs[key] = v/max(c, 1e-6)
        for key in self.value_ep:
            if key not in self.counter_ep:
                mean_vs[key] = self.value_ep[key]
        self.counter_ep = {}
        self.value_ep = {}
        self.last_mean_vs = mean_vs
        self.epoch = epoch
        return mean_vs

    def add_scalar_ep(self, name, value, epoch):
        '''
        call this method when you don't want to calculate value means at ends of epochs.
        '''
        if self.writer is not None:
            self.writer.add_scalar('ep/' + name, value, epoch)
        self.value_ep[name] = value

    def state_dict(self):
        return {'counter_ep' : self.counter_ep,
                'counter_tot' : self.counter_tot,
                'value_ep' : self.value_ep,
                'last_mean_vs' : self.last_mean_vs,
                'epoch' : self.epoch}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)
