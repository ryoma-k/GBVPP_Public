import argparse
import json
from pathlib import Path
from pprint import pprint
import random
import sys
import torch
import yaml
from torch.utils.data import DataLoader
from base.trainers import DL_sample
from base.model import MyDataParallel
from lstm import GBVSSLSTMModel, GBVSSLSTMTrainer, GBVPPEvaluator, make_datasets, preprocess_datasets
from base.utils import seed_everything, Tools as T, Summary


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        usage='python main.py -cp yamls/base.yaml',
        add_help=True
    )
    parser.add_argument('-cp', '--config_path', type=str, nargs='+',
                        default=['config.yaml'], help='config path')
    parser.add_argument('-seed', '--seed', type=int)
    parser.add_argument('-nw', '--num_worker', type=int)
    parser.add_argument('-memo', '--memo', type=str)
    parser.add_argument('-multi', '--multi_gpu', action='store_true')
    parse = parser.parse_args()

    # load yaml
    config = {}
    config_names = []
    for config_path in parse.config_path:
        config_names.append(Path(config_path).name.split('.')[0])
        with open(config_path) as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        config = T.dict_merge(config, cfg)
    config_name = '_'.join(config_names)

    # fix seed
    seed = parse.seed
    if seed is None: seed = config['main'].get('seed')
    if seed is None: seed = random.randint(1, 10000)
    if seed is not None: config_file_name_seed = config_name + f'_seed_{seed}'
    seed_everything(seed)
    if parse.memo is not None:
        config_file_name_seed += ('_' + parse.memo)
    config['main']['save_path'] += ('/' + config_file_name_seed)
    config['main']['seed'] = seed
    config['trainer']['save_path'] = config['main']['save_path']
    config['dataset']['save_path'] = config['main']['save_path']
    pprint(config)
    T.make_dir(config['main']['save_path'] + '/pths')
    with open(config['main']['save_path'] + '/config.yaml', 'w') as f:
        yaml.dump(config, f)

    # tensorboard logger
    summary = Summary(config['main']['save_path'])
    try:
        summary.add_text('hash', T.get_git_hash())
        summary.add_text('diff', T.get_git_diff())
    except:
        print(".git doesn't exist.")
    summary.add_text('config', json.dumps(config))

    # make dataset & dataloader
    if parse.num_worker is None:
        num_w = 0 if config['main'].get('debug', False) else 4
    else:
        num_w = parse.num_worker
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    trn_bs = config['main']['trn_bs']
    val_bs = config['main'].get('val_bs', trn_bs)
    if config['main']['preprocess']:
        with T.timer('data load'):
            preprocess_datasets(config['dataset'], seed=seed)
        sys.exit(0)
    else:
        with T.timer('data load'):
            trn_dss, val_dss = make_datasets(config['dataset'], seed=seed)
    trn_dls = []
    for trn_ds, bs, nw in zip(trn_dss, [trn_bs, config['main'].get('pseudo_bs')], [num_w, 2]):
        trn_dls.append(DL_sample(DataLoader(trn_ds, batch_size=bs, shuffle=True,
                                 num_workers=nw, drop_last=True)))
    val_dls = {}
    for key in val_dss:
        val_dl = DataLoader(val_dss[key], batch_size=bs, shuffle=False, num_workers=num_w)
        val_dls[f'val{key}'] = val_dl

    dataloaders = {'trn': trn_dls,
                   'tst': val_dls}

    # make model, loss
    if config['model']['model_type'] == 'lstm':
        model = GBVSSLSTMModel(config['model']).cuda()
    else:
        raise Exception
    if parse.multi_gpu:
        model = MyDataParallel(model)
    evaluator = GBVPPEvaluator(**config['loss'], summary=summary)

    # make trainer
    trainer = GBVSSLSTMTrainer(model, dataloaders, evaluator, summary)
    trainer.set_args(config['trainer'])

    # resume option
    with T.timer('training'):
        if config['main']['resume'] is not None:
            pths = torch.load(config['main']['resume'])
            if config['main'].get('only_model_load'):
                print('load only model...')
            trainer.load_resume(pths, config['main'].get('only_model_load', False))
            if config['main'].get('reset_lr'):
                trainer.set_lr(config['main']['reset_lr'])
        try:
            with torch.autograd.set_detect_anomaly(True):
                # training
                trainer.run()
            trainer.get_state_pth(trainer.iteration)
        except KeyboardInterrupt:
            torch.cuda.empty_cache()
            trainer.get_state_pth(f'{trainer.iteration}_interrupt')
            print('##### interrupt #####')
