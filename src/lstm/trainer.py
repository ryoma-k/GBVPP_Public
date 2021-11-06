import torch
from tqdm import tqdm
from base.trainers import Trainer


class GBVSSLSTMTrainer(Trainer):
    def lstm_iter(self, data, stage='trn'):
        inputs = self.data2device(data, ['cate_v', 'cont_v'])
        labels = self.data2device(data, ['label', 'mask', 'cate_v', 'u_out'])
        outputs = self.model(**inputs)
        with torch.no_grad():
            self.evaluator.criterion(outputs, labels, stage=stage)
        with torch.set_grad_enabled(stage == 'trn'):
            loss = self.evaluator(outputs, labels, mode=self.loss_mode, stage=stage)
        return outputs, loss

    def set_args(self, args):
        self.state_dict.update(args)
        self.set_optimizer()
        self.set_scheduler()
        if self.state_dict['model_type'] == 'lstm':
            setattr(self, 'run_iter_trn', self.lstm_iter)
            setattr(self, 'run_iter_tst', self.lstm_iter)
        else:
            raise Exception
        self.loss_mode = f'masked_{self.state_dict["loss_mode"]}'

    def run_test(self):
        self.model.eval()
        dl = self.dataloaders['tst']
        outputs = []
        with torch.no_grad():
            for data in tqdm(dl, leave=False):
                breath_id = data['breath_id']
                output, _ = self.run_iter_tst(data, stage='tst')
                outputs.append((breath_id, output[...,0].cpu()))
        return outputs
