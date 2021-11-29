import numpy as np
import argparse
from modify_main import main_cgcnn
from modify_predict import predict_cgcnn
import torch
import os
from modify_data import CIFData


CO_HYPERPARAMS = {
    'batch_size': 214,
    'lr': 5.6e-3,
    'epochs': 150,
    'atom_fea_len': 46,
    'h_fea_len': 83,
    'n_conv': 8,
    'n_h': 4
}
H_HYPERPARAMS = {
    'batch_size': 140,
    'lr': 1e-3,
    'epochs': 150,
    'atom_fea_len': 107,
    'h_fea_len': 50,
    'n_conv': 6,
    'n_h': 1
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adsorbate', type=str, choices=['co', 'h'], default='co')
    parser.add_argument('--optim', type=str, choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--atom_fea_len', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument('--epochs', type=int, default=-1)
    parser.add_argument('--h_fea_len', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--lr_milestones', nargs='+', default=[100])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--n_conv', type=int, default=-1)
    parser.add_argument('--n_h', type=int, default=-1)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--task', default='regression')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--root_dir')

    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    
    files = os.listdir(args.root_dir)
    data_size = len([f for f in files if f.endswith('.cif')])
    args.train_size, args.val_size = round(data_size * 0.6),  round(data_size*0.2)
    args.test_size = data_size - args.train_size - args.val_size
    args.cuda = not args.disable_cuda

    if args.adsorbate == 'co':
        for name, value in CO_HYPERPARAMS.items():
            setattr(args, name, value)
    elif args.adsorbate == 'h':
        for name, value in H_HYPERPARAMS.items():
            setattr(args, name, value)
    else:
        raise NotImplementedError("co or h only")

    dataset = CIFData(args.root_dir)

    epoches, train_mae_errors, train_losses, val_mae_errors, val_losses, test_mae, test_loss = main_cgcnn(args, dataset=dataset)
