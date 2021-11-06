from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler as lrs
from tqdm import tqdm


class Trainer():
    def __init__(self, model, dataloaders, evaluator, summary, save=True):
        self.model = model
        self.device = next(iter(self.model.parameters())).device
        self.dataloaders = dataloaders
        self.evaluator = evaluator.to(self.device)
        self.summary = summary
        self.save = save
        self.state_dict = dict()
        self.iteration, self.ini_iter, self.epoch = 0, 0, 0

    def set_args(self, args):
        self.state_dict.update(args)
        self.set_optimizer()
        self.set_scheduler()
        setattr(self, 'run_iter_trn', self.classify_iter)
        setattr(self, 'run_iter_tst', self.classify_iter)

    def classify_iter(self, data, stage='trn'):
        image = data['img'].to(self.device).float()
        label = data['label'].to(self.device).float()
        output = self.model(image)
        loss = self.evaluator(output, label, mode='class', stage=stage)
        with torch.no_grad():
            self.criterion(output, label, stage=stage)
        return output, loss

    def run(self):
        data = None
        for iteration in tqdm(range(self.ini_iter, self.state_dict['iterations']),
                              dynamic_ncols=True, smoothing=0.01):
            self.iteration = iteration
            # TRAIN
            self.model.train()
            with torch.set_grad_enabled(True):
                if self.state_dict['scheduler'] != 'lr_find' or data is None:
                    data = self.load_data()
                output, loss = self.run_iter_trn(data, stage='trn')
                self.optimizer[0].zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.state_dict['max_grad_norm'])
                self.summary.add_scalar('grad_norm', grad_norm)
                self.optimizer[0].step()
                if len(self.scheduler) > 0:
                    self.scheduler[0].step()
                    if self.summary is not None:
                        self.summary.add_scalar('lr', self.scheduler[0].get_last_lr()[-1], take_average=False)
            if hasattr(self.model, 'update_ema'):
                self.model.update_ema()
            # TEST
            if iteration != 0 and (iteration+1) % self.state_dict['test_freq'] == 0:
                self.model.eval()
                with torch.no_grad():
                    dls = self.dataloaders['tst']
                    for key in dls:
                        for data in tqdm(dls[key], leave=False):
                            output, loss = self.run_iter_tst(data, stage=key)
                self.end_epoch()
                if self.summary is not None:
                    self.summary.end_epoch(self.epoch)
                if self.evaluator.is_best_score():
                    self.get_state_pth('best')
            if iteration != 0 and (iteration+1) % self.state_dict.get('save_freq', 5000) == 0:
                self.get_state_pth(iteration)

    def load_data(self, stage='trn'):
        data = {}
        dl_datas = [next(dl) for dl in self.dataloaders[stage]]
        if len(dl_datas) == 1: data = dl_datas[0]
        else:
            keys = dl_datas[0].keys()
            for key in keys:
                data[key] = torch.cat([dl_data[key] for dl_data in dl_datas], 0)
        return data

    def data2device(self, data, keys):
        return {key: data[key].to(self.device) for key in keys}

    def set_optimizer(self, optim_state=None):
        if optim_state is None:
            optim_state = self.state_dict['optim_state']
        parameters = self.model.get_parameters(lr=optim_state.get('lr'))
        self.optimizer = []
        for param in parameters:
            if hasattr(optim, self.state_dict['optimizer']):
                self.optimizer.append(eval(f'optim.{self.state_dict["optimizer"]}(param, **optim_state)'))
            else:
                raise Exception

    def set_scheduler(self, scheduler_state=None):
        if self.state_dict['scheduler'] is None:
            self.scheduler = []
            return
        if scheduler_state is None:
            scheduler_state = self.state_dict.get('scheduler_state', {})
        self.scheduler = []
        for optimizer in self.optimizer:
            if self.state_dict['scheduler'] == 'lr_find':
                self.scheduler.append(lrs.LambdaLR(optimizer, lr_lambda=lambda x : np.exp(x/10)))
            elif self.state_dict['scheduler'] == 'Lambda':
                lr_lambda = scheduler_state.pop('lr_lambda')
                self.scheduler.append(lrs.LambdaLR(optimizer, **scheduler_state,
                                      lr_lambda=[eval(lr_lam) for lr_lam in lr_lambda]))
            else:
                scheduler = eval(f'lrs.{self.state_dict["scheduler"]}')
                self.scheduler.append(scheduler(optimizer, **scheduler_state))
        return

    def end_epoch(self):
        self.evaluator.end_epoch(self.epoch)
        self.epoch += 1

    def get_state_pth(self, name, save=True, state_dict=None):
        if state_dict is None:
            state_dict = {'model': self.model.state_dict(),
                          'optimizer': [opt.state_dict() for opt in self.optimizer],
                          'scheduler': [sch.state_dict() for sch in self.scheduler],
                          'summary': self.summary.state_dict() if self.summary is not None else None,
                          'iteration': self.iteration,
                          'epoch': self.epoch}
        if save:
            torch.save(state_dict, Path(self.state_dict['save_path']) / 'pths' / f'{name}.pth')
        return state_dict

    def load_resume(self, pths, only_model_load=False):
        self.model.load_state_dict(pths['model'])
        if only_model_load: return
        for i in range(len(self.optimizer)):
            self.optimizer[i].load_state_dict(pths['optimizer'][i])
        for i in range(len(self.scheduler)):
            self.scheduler[i].load_state_dict(pths['scheduler'][i])
        if self.summary is not None:
            self.summary.load_state_dict(pths['summary'])
        self.ini_iter = pths['iteration'] + 1
        self.epoch = pths['epoch']

    def set_lr(self, lr, set_initial=True):
        for optimizer in self.optimizer:
            for param in optimizer.param_groups:
                param['lr'] = lr
                if set_initial: param['initial_lr'] = lr
        self.set_scheduler()
