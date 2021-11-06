from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset
from base.utils import Tools as T


def preprocess_datasets(ds_config, seed):
    # for data preprocess. called when yamls/preprocess.yaml is used
    root = Path(ds_config['root'])
    train_csv_path = ds_config['train_csv_path']
    test_csv_path = ds_config['test_csv_path']
    with T.timer('csv load'):
        trn_val_df = pd.read_csv(root/train_csv_path)
        test_df = pd.read_csv(root/test_csv_path)
    r_map = {5: 0, 20: 1, 50: 2}
    c_map = {10: 0, 20: 1, 50: 2}
    trn_val_df['R'] = trn_val_df['R'].map(r_map)
    trn_val_df['C'] = trn_val_df['C'].map(c_map)
    test_df['R'] = test_df['R'].map(r_map)
    test_df['C'] = test_df['C'].map(c_map)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    bids = trn_val_df.breath_id.unique()
    for fold_id, (_, val_idx) in enumerate(kfold.split(range(len(bids)))):
        val_bids = bids[val_idx]
        trn_val_df.loc[trn_val_df.breath_id.isin(val_bids), 'split'] = fold_id
    for fold_id in range(10):
        print(f'{fold_id} : {len(trn_val_df[trn_val_df.split == fold_id])}   ', end='')
    print(f'total : {trn_val_df.split.isin(range(10)).sum()} / {len(trn_val_df)}')
    trn_val_df.to_csv(root/ds_config['train_load_path'], index=False)
    test_df.to_csv(root/ds_config['test_load_path'], index=False)


def make_datasets(ds_config, seed, train=True):
    # make datasets for training/submitting
    root = Path(ds_config['root'])
    if train:
        csv_path = ds_config['train_load_path']
        split = ds_config['split']
        spkey = 'split'
        print(f'### using SPLIT {split} ###')
        with T.timer('csv load'):
            trn_val_df = pd.read_csv(root/csv_path)
        trn_df = trn_val_df[~trn_val_df[spkey].isin(split)]
        trn_datasets = [GBVPPDataset(trn_df, ds_config)]
        if ds_config.get('pseudo_load_path'):
            print('### using PSEUDO LABEL ###')
            pseudo_df = pd.read_csv(ds_config['pseudo_load_path'])
            if ds_config.get('pseudo_split'):
                ps_split = ds_config['pseudo_split']
                print(f'### using PSEUDO SPLIT {ps_split} ###')
                pseudo_df = pseudo_df[~pseudo_df['split'].isin(ps_split)]
            trn_datasets.append(GBVPPDataset(pseudo_df, ds_config))
        for trn_dataset, key in zip(trn_datasets, ['trn', 'pseudo']):
            print(f'{key} size : {len(trn_dataset)}')
        val_datasets = {}
        for sp in (*split, split):
            key, sp = (f'sp_{sp}', [sp]) if isinstance(sp, int) else ('', sp)
            val_df = trn_val_df[trn_val_df[spkey].isin(sp)]
            val_dataset = GBVPPDataset(val_df, ds_config)
            print(f'val split{sp} size : {len(val_dataset)}')
            val_datasets[key] = val_dataset
        return trn_datasets, val_datasets
    else:
        csv_path = ds_config['test_load_path']
        with T.timer('csv load'):
            test_df = pd.read_csv(root/csv_path)
        if ds_config.get('pseudo_split'):
            split = ds_config['pseudo_split']
            print(f'### using PSEUDO SPLIT {split} ###')
            test_df = test_df[test_df['split'].isin(split)]
        tst_dataset = GBVPPDataset(test_df, ds_config, train=False)
        print(f'tst size : {len(tst_dataset)}')
        return tst_dataset


class GBVPPDataset(Dataset):
    # Vent dataset
    def __init__(self, df, ds_config, train=True):
        self.df = df.sort_values(['breath_id', 'id']).reset_index(drop=True)
        self.config = ds_config
        self.groups = self.df.groupby('breath_id').groups
        self.keys = sorted(list(self.groups.keys()))
        self.train = train
        self.length = ds_config.get('length', 80)
        self.skip = ds_config.get('skip', 1)
        self.cate_vs = self.df[self.config['cate_cols']].values
        self.cont_vs = self.df[self.config['cont_cols']].values
        self.u_outs = self.df['u_out'].values
        if 'th_mask' in self.df.columns:
            # for pseudo-label phase
            self.th_mask = self.df['th_mask'].values
        else:
            self.th_mask = np.ones_like(self.u_outs).astype('bool')
        if 'pressure' in self.df.columns:
            self.pressure = self.df['pressure'].values

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        output = {}
        output['index'] = idx
        output['breath_id'] = self.keys[idx]
        indexes = self.groups[self.keys[idx]]
        output['cate_v'] = torch.LongTensor(self.cate_vs[indexes])[:self.length][::self.skip]
        output['cont_v'] = torch.FloatTensor(self.cont_vs[indexes])[:self.length][::self.skip].log1p()
        output['u_out'] = torch.LongTensor(self.u_outs[indexes] > 0)[:self.length][::self.skip]
        output['mask'] = (output['u_out'] == 0) & (torch.BoolTensor(self.th_mask[indexes][:self.length][::self.skip]))
        if self.train:
            output['label'] = torch.FloatTensor(self.pressure[indexes])[:self.length][::self.skip]
        else:
            output['label'] = torch.FloatTensor([0] * (self.length // self.skip))
        return output
