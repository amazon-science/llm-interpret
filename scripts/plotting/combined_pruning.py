import os
import json
import torch
import argparse
import numpy as np
from style import *

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str, default=None)
parser.add_argument("--save_plot_path", type=str, default=None)
parser.add_argument("--shot", type=str, default="0-shot", choices=["0-shot","1-shot","5-shot"])

args = parser.parse_args()

fc_percents   = [0, 10, 20, 30, 40, 50]
head_percents = [0, 10, 20, 30, 40, 50, 60, 70]

def get_accuracy(path, task):
    with open(path) as f:
        x = json.load(f)
    return x['results'][f'{task}']['acc'] * 100 if task != "record" else x['results'][f'{task}']['em'] * 100
    

datasets = ['hellaswag','piqa', 'arc_easy', 'arc_challenge', 'openbookqa', 'winogrande', 'boolq', 'cb', 'copa', 'wic', 'wsc', 'multirc', 'rte', 'record']
prefix = "1shot_" if '1-shot' in args.shot else "5shot_" if '5-shot' in args.shot else ""

matrix = []
for dataset in datasets:
    # files = os.listdir(os.path.join(args.results_path, dataset))
    # num_files_fc = list(filter(lambda x: '_fc_percent.txt' in x, files))
    # num_files_head = list(filter(lambda x: '_percent.txt' in x and len(x) == 14, files))
    # num_files_head_fc = list(filter(lambda x: '_head_percent.txt' in x, files))
    # if len(num_files_fc) == 9 and len(num_files_head) == 11 and len(num_files_head_fc) == 35 and 'none.txt' in files:
    print(dataset)
    performance_matrix = np.zeros((len(head_percents), len(fc_percents)))
    for fc_percent in fc_percents:
        for head_percent in head_percents:
            if fc_percent == 0 and head_percent == 0:
                performance_matrix[0][0] = get_accuracy(os.path.join(args.results_path, dataset, f'{prefix}0_percent.txt'), dataset)
            elif fc_percent == 0:
                performance_matrix[head_percent//10][0] = get_accuracy(os.path.join(args.results_path, dataset, f'{prefix}{head_percent}_percent.txt'), dataset)
            elif head_percent == 0:
                performance_matrix[0][fc_percent//10] = get_accuracy(os.path.join(args.results_path, dataset, f'{prefix}{fc_percent}_fc_percent.txt'), dataset)
            else:
                performance_matrix[head_percent//10][fc_percent//10] = get_accuracy(os.path.join(args.results_path, dataset, f'{prefix}{fc_percent}_fc_{head_percent}_head_percent.txt'), dataset)
    matrix.append(np.expand_dims(performance_matrix, axis = 0))
    max_, min_ = np.amax(performance_matrix), np.amin(performance_matrix)
    ax = sns.heatmap(performance_matrix, xticklabels = fc_percents, yticklabels = head_percents, cmap="YlGnBu", annot = True, vmax = max_, vmin = min_)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.title(f'Performance on {dataset} with Removal of Heads + FFN')
    plt.xlabel('Pruning of FFN (%)')
    plt.ylabel('Pruning of Attention Heads (%)')
    os.makedirs(os.path.dirname(args.save_plot_path), exist_ok = True)
    plt.savefig(os.path.join(args.save_plot_path, f'{dataset}.png'))
    plt.savefig(os.path.join(args.save_plot_path, f'{dataset}.pdf'))
    plt.close()

matrix = np.concatenate(matrix, axis = 0)
matrix = np.mean(matrix, axis = 0)
max_, min_ = np.amax(matrix), np.amin(matrix)
ax = sns.heatmap(matrix, xticklabels = fc_percents, yticklabels = head_percents, cmap="YlGnBu", annot = True, vmax = max_, vmin = min_)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.title(f'Average Performance after Combined Pruning of Heads and FFNs ({args.shot})')
plt.xlabel('Pruning of FFNs (%)')
plt.ylabel('Pruning of Attention Heads (%)')
os.makedirs(os.path.dirname(args.save_plot_path), exist_ok = True)
plt.savefig(os.path.join(args.save_plot_path, f'averaged.png'))
plt.savefig(os.path.join(args.save_plot_path, f'averaged.pdf'))
plt.close()
