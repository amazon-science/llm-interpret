import os
import argparse
import pickle
import torch
import numpy as np
from style import *

parser = argparse.ArgumentParser()
parser.add_argument("--saved_head_importance_path", type=str, default=None)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--save_plot_path", type=str, default=None)
parser.add_argument("--aggregate", action = "store_true")
parser.add_argument("--shot", type=str, default="0-shot", choices=["0-shot","1-shot","5-shot"])

args = parser.parse_args()

prefix = "1shot_" if '1-shot' in args.shot else "5shot_" if '5-shot' in args.shot else ""

if args.aggregate:
    datasets = ['hellaswag','piqa', 'arc_easy', 'arc_challenge', 'openbookqa', 'winogrande', 'boolq', 'cb', 'copa', 'wic', 'wsc', 'multirc', 'rte', 'record']
    himp = []
    for dataset in datasets:
        pth = f'{dataset}.pkl' if "0-shot" in args.shot else f'1shot_{dataset}.pkl' if "1-shot" in args.shot else f'5shot_{dataset}.pkl'
        file_path = os.path.join(args.saved_head_importance_path, pth)
        print(file_path)
        with open(file_path, 'rb') as handle:
            himp.append(pickle.load(handle).unsqueeze(0))
    results = torch.mean(torch.cat(himp, dim = 0), dim = 0)
    with open(os.path.join(os.path.dirname(args.saved_head_importance_path),f'{prefix}aggregate.pkl'), 'wb') as f:
        pickle.dump(results, f)
        print('aggregate saved!')
else:
    with open(args.saved_head_importance_path, 'rb') as handle:
        results = pickle.load(handle)


results = results.view(64, 72)
layers, heads = results.shape
min_, max_ = torch.min(results).item(), torch.max(results).item() 
if args.aggregate:
    ax = sns.heatmap(results, xticklabels = [(i+1) if i%2==0 else None for i in range(heads)], yticklabels = [(i+1)if i%2==0 else None for i in range(layers)], vmax = 0.002 if '1-shot' in args.shot else 0.004 if '0-shot' in args.shot else 0.0014)
else:
    ax = sns.heatmap(results, xticklabels = [(i+1) if i%2==0 else None for i in range(heads)], yticklabels = [(i+1)if i%2==0 else None for i in range(layers)], vmin = min_ , vmax = max_/10)
if not args.aggregate:
    dataset_name = DATASET_TO_OFFICIAL[args.dataset]
else:
    dataset_name = 'Aggregate'
plt.title(f'Head Importance Score ({args.shot} {dataset_name})') # title with fontsize 20
plt.xlabel("Heads") # x-axis label with fontsize 15
plt.ylabel("Layers") # y-axis label with fontsize 15
ax.invert_yaxis()


os.makedirs(os.path.dirname(args.save_plot_path), exist_ok = True)
plt.savefig(args.save_plot_path)