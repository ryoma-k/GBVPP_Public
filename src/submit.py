import argparse
import os
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import yaml
from torch.utils.data import DataLoader
from lstm import GBVSSLSTMModel, GBVSSLSTMTrainer, GBVPPEvaluator, make_datasets
from base.utils import Tools as T


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        usage='python submit.py -cp ../result/base_seed_0/config.yaml x N-files '
              '-pths ../result/base_seed_0/pths/best.pth x N-files',
        add_help=True
    )
    parser.add_argument('-cp', '--config_path', type=str, nargs='+',
                        default=['config.yaml'], help='config path')
    parser.add_argument('-seed', '--seed', type=int)
    parser.add_argument('-nw', '--num_worker', type=int)
    parser.add_argument('-memo', '--memo', type=str)
    parser.add_argument('-pths', '--pth_paths', type=str, nargs='+')
    parser.add_argument('-ent', '--ensemble_type', type=str, default='median')
    parser.add_argument('-multi', '--multi_gpu', action='store_true')
    parser.add_argument('-sub', '--submit', action='store_true')
    parser.add_argument('-no_pp', '--no_post_process', action='store_true')
    parser.add_argument('-fn', '--file_name', type=str)
    parser.add_argument('-pseudo', '--for_pseudo', action='store_true')
    parser.add_argument('-val', '--validation', action='store_true')
    parser.add_argument('-ps_th', '--pseudo_threshold', type=float, default=float('inf'))
    parser.add_argument('-clip_N', '--clip_N', type=int, default=3)
    parser.add_argument('-nsp', '--n_splits', type=int)
    parser.add_argument('-stack', '--for_stacking', action='store_true')
    parse = parser.parse_args()

    nw = 6
    val_bs = 128
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    breath_ids = []
    predicts = []
    for config_path, pth_path in zip(parse.config_path, parse.pth_paths):
        # load yaml
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        config['trainer']['save_path'] = config['main']['save_path']

        # make dataset & dataloader
        with T.timer('data load'):
            if parse.validation:
                tst_ds = make_datasets(config['dataset'], seed=0)[1]['']
            else:
                tst_ds = make_datasets(config['dataset'], seed=0, train=False)
        tst_dl = DataLoader(tst_ds, batch_size=val_bs, shuffle=False, num_workers=nw)
        dataloaders = {'tst': tst_dl}

        # make model, loss
        if config['model']['model_type'] == 'lstm':
            model = GBVSSLSTMModel(config['model']).cuda()
        else:
            raise Exception
        evaluator = GBVPPEvaluator(**config['loss'], summary=None)

        # make trainer
        trainer = GBVSSLSTMTrainer(model, dataloaders, evaluator, None)
        trainer.set_args(config['trainer'])
        trainer.load_resume(torch.load(pth_path), config['main'].get('only_model_load', False))

        # TEST
        with T.timer('testing'):
            output = trainer.run_test()
            breath_id = torch.cat([out[0].view(-1) for out in output], -1).numpy()
            predict = torch.cat([out[1].view(-1) for out in output], -1).numpy()
        breath_ids.append(breath_id)
        predicts.append(predict.reshape(-1, 80))
    breath_ids = np.concatenate(breath_ids, 0)
    predicts = np.concatenate(predicts, 0)
    n_bid = len(np.unique(breath_ids))
    predicts = predicts[breath_ids.argsort(axis=0)].reshape(n_bid, -1, 80).transpose(1,0,2).reshape(-1, n_bid*80)

    if parse.for_stacking:
        tst_df = pd.read_csv('../input/ventilator-pressure-prediction/train_process.csv')
    elif parse.for_pseudo:
        predicts_cp = predicts.copy()
        predicts_cp = predicts_cp.reshape(predicts.shape[0], -1, 80)
        N = parse.clip_N
        predicts_cp = np.sort(predicts_cp, axis=0)[N:len(predicts_cp)-N]
        pseudo_mask = (predicts_cp[-1] - predicts_cp[0] < parse.pseudo_threshold).reshape(-1)
        tst_df['th_mask'] = pseudo_mask
        print(f'{pseudo_mask[tst_ds.u_outs == 0].mean()} % used for pseudo')
    else:
        tst_df = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')

    # ensemble
    if parse.ensemble_type == 'mean':
        predicts = predicts.mean(0)
    elif parse.ensemble_type == 'median':
        predicts = np.median(predicts, 0)

    if parse.for_stacking:
        tst_df['stacking'] = predicts
    else:
        tst_df['pressure'] = predicts

    # post process
    if not parse.no_post_process:
        print('POST PROCESSING ...')
        trn_val_df = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
        unique_pressures = trn_val_df["pressure"].unique()
        sorted_pressures = np.sort(unique_pressures)
        total_pressures_len = len(sorted_pressures)

        def find_nearest(prediction):
            insert_idx = np.searchsorted(sorted_pressures, prediction)
            if insert_idx == total_pressures_len:
                return sorted_pressures[-1]
            elif insert_idx == 0:
                return sorted_pressures[0]
            lower_val = sorted_pressures[insert_idx - 1]
            upper_val = sorted_pressures[insert_idx]
            return lower_val if abs(lower_val - prediction) < abs(upper_val - prediction) else upper_val

        tst_df['pressure'] = tst_df['pressure'].apply(find_nearest)

    # split
    if parse.n_splits is not None:
        kfold = KFold(n_splits=parse.n_splits, shuffle=True, random_state=0)
        bids = tst_df.breath_id.unique()
        for fold_id, (_, val_idx) in enumerate(kfold.split(range(len(bids)))):
            val_bids = bids[val_idx]
            tst_df.loc[tst_df.breath_id.isin(val_bids), 'split'] = fold_id

    # preserve
    if parse.file_name is None:
        name = [os.path.splitext(pth_path)[0].split('/') for pth_path in parse.pth_paths]
        name = '__'.join([x[-3] + '_' + x[-1] for x in name])
    else:
        name = parse.file_name
    print(f'save as {name}.csv')

    root_dir = 'pseudos' if parse.for_pseudo else 'submits'
    T.make_dir(f'../{root_dir}')
    tst_df.to_csv(f'../{root_dir}/{name}.csv', index=False)

    # submit to kaggle
    if parse.submit:
        message = name
        if parse.memo is not None:
            message = message + f'\n{parse.memo}'
        subprocess.call(f'kaggle competitions submit ventilator-pressure-prediction '\
                        f'-f ../submits/{name}.csv -m "{message}"', shell=True)
    print()
