from torch.utils.data import Dataset


class DL_sample():
    '''
    this sampler doesn't stop at its end of iteration.
    '''
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = self.dataloader.dataset
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.reset()
            return next(self.iterator)

    def reset(self):
        self.iterator = iter(self.dataloader)


class SimpleDataset(Dataset):
    def __init__(self, imgs, labels, transforms=None, **args):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms
        self.keys = []
        for key in args:
            setattr(self, key, args[key])
            self.keys.append(key)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        data = {}
        data['index'] = index
        data['img'] = self.imgs[index]
        if self.transforms is not None:
            data['img'] = self.transforms(data['img'])
        data['label'] = self.labels[index]
        for key in self.keys:
            obj = getattr(self, key)
            data[key] = obj[index] if hasattr(obj, '__getitem__') else obj
        return data
