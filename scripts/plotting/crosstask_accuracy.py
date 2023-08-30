import os
import torch
import json
import random
import argparse
import numpy as np
from style import *

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str, default=None)
parser.add_argument("--save_plot_path", type=str, default=None)
parser.add_argument("--shot", type=str, default="0-shot", choices=["0-shot","1-shot","5-shot"])
parser.add_argument("--ood", action='store_true')

args = parser.parse_args()

## edit this to change the dataset
datasets = ['record']

cross_task_map = {'0-shot':{'copa':{'high':'winogrande', 'low': 'record'},'winogrande':{'high':'copa', 'low':'record'},
                'record':{'high':'rte', 'low': 'openbookqa'}},
                '1-shot':{'copa':{'high':'wsc', 'low': 'record'},'winogrande':{'high':'copa', 'low':'record'},
                'record':{'high':'hellaswag', 'low': 'wic'}},
                '5-shot':{'copa':{'high':'wsc', 'low': 'record'},'winogrande':{'high':'boolq', 'low':'record'},
                'record':{'high':'hellaswag', 'low': 'wic'}}}

files = [10, 30, 50, 70, 90]
prefix = "1shot_" if '1-shot' in args.shot else "5shot_" if '5-shot' in args.shot else ""

for dataset in datasets:
    original_path = os.path.join(args.results_path, dataset)
    l_n, l_h, l_l, l_a = [], [], [], []
    for fname in files:
        filename_normal = os.path.join(original_path, f'{prefix}{fname}_percent.txt')
        if not args.ood:
            filename_high = os.path.join(args.results_path, 'cross_task', f'{prefix}{dataset}_high', f'{fname}_percent.txt')
            filename_low = os.path.join(args.results_path, 'cross_task', f'{prefix}{dataset}_low', f'{fname}_percent.txt')
        filename_agg = os.path.join(args.results_path, 'cross_task', f'{prefix}{dataset}_aggregate', f'{fname}_percent.txt')
        metric = 'em' if dataset=='record' else 'acc'
        with open(filename_normal, 'rb') as f:
            res = json.load(f)
            l_n.append(res['results'][f'{dataset}'][metric] * 100)
        if not args.ood:
            with open(filename_high, 'rb') as f:
                res = json.load(f)
                l_h.append(res['results'][f'{dataset}'][metric] * 100)
            with open(filename_low, 'rb') as f:
                res = json.load(f)
                l_l.append(res['results'][f'{dataset}'][metric] * 100)
        with open(filename_agg, 'rb') as f:
            res = json.load(f)
            l_a.append(res['results'][f'{dataset}'][metric] * 100)
    plt.plot(files, l_n, marker = 'o', label = f'Using {DATASET_TO_OFFICIAL[dataset]} ranking ({args.shot})')
    if not args.ood:
        high_dataset, low_dataset = cross_task_map[args.shot][dataset]['high'], cross_task_map[args.shot][dataset]['low']
        plt.plot(files, l_h, marker = 'o', label = f'Using {DATASET_TO_OFFICIAL[high_dataset]} dataset ranking ({args.shot})')
        plt.plot(files, l_l, marker = 'o', label = f'Using {DATASET_TO_OFFICIAL[low_dataset]} dataset ranking ({args.shot})')
    plt.plot(files, l_a, marker = 'o', label = f'Using Aggregate Ranking ({args.shot})')
    plt.legend()
    plt.xlabel('Percentage Pruned (%)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Cross Task Transfer for {DATASET_TO_OFFICIAL[dataset]}')
    plt.grid()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.save_plot_path), exist_ok = True)
    plt.savefig(os.path.join(args.save_plot_path, f'{prefix}{dataset}.png'))
    plt.savefig(os.path.join(args.save_plot_path, f'{prefix}{dataset}.pdf'))
    plt.close()