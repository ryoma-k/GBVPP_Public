import torch.nn as nn


class Criterion(nn.Module):
    def __init__(self, summary=None):
        super().__init__()
        self.summary = summary

    def __call__(self, output, label, stage):
        acc = self.accuracy(output, label)
        if self.summary is not None:
            self.summary.add_scalar(f'{stage}/accuracy', acc)

    def accuracy(self, output, label, ignore=-100):
        return (output.argmax(-1) == label)[label != ignore].float().mean()

    def end_epoch(self, epoch):
        pass

    def is_best_score(self):
        return False
