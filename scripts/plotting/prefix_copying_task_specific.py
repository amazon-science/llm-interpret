import os
import torch
import pickle
import argparse
import numpy as np
from style import *

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str, default=None)
parser.add_argument("--shot", type=str, default="0-shot", choices=["0-shot","1-shot","5-shot"])
parser.add_argument("--prefix_matching_path", type=str, default=None)
parser.add_argument("--copying_path", type=str, default=None)
parser.add_argument("--save_prefix_plot_path", type=str, default=None)
parser.add_argument("--save_copying_plot_path", type=str, default=None)

args = parser.parse_args()


datasets = ['hellaswag','piqa', 'arc_easy', 'arc_challenge', 'openbookqa', 'winogrande', 'boolq', 'cb', 'copa', 'wic', 'wsc', 'multirc', 'rte', 'record']
prefix = "1shot_" if '1-shot' in args.shot else "5shot_" if '5-shot' in args.shot else ""


def open_file(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res

prefix_matching_scores = open_file(args.prefix_matching_path)['mean'].view(-1)
copying_scores = open_file(args.copying_path)['mean'].view(-1)
total_prefix = prefix_matching_scores.sum()
total_copying = copying_scores.sum()

# shots = [0, 1, 5]
# aggregate_paths = [args.aggregate_path_0_shot, args.aggregate_path_1_shot, args.aggregate_path_5_shot]

for dataset in datasets:
    rank_path = os.path.join(args.results_path, f'{prefix}{dataset}.pkl')
    aggregate_scores = open_file(rank_path).view(-1)
    assert aggregate_scores.shape == prefix_matching_scores.shape == copying_scores.shape
    _, ranking = torch.sort(aggregate_scores)
    sum_prefix = []
    x_axis = []
    for i in range(len(ranking)):
        sum_prefix.append((prefix_matching_scores[ranking[i:]].sum().item() * 100) / total_prefix)
        x_axis.append((100 * i) / len(ranking))
    plt.plot(x_axis, sum_prefix, label = DATASET_TO_OFFICIAL[dataset], color=DATASET_TO_COLOR[dataset])

plt.xlabel('Percentage pruned (%)', fontsize=20)
plt.ylabel('% of Total Prefix Matching Score Retained', fontsize=20)
plt.title(f'Impact of Pruning Attention Heads on Prefix Matching ({args.shot})', wrap=True)
plt.legend()
plt.grid()
os.makedirs(os.path.dirname(args.save_prefix_plot_path), exist_ok = True)
plt.savefig(args.save_prefix_plot_path)
plt.savefig(args.save_prefix_plot_path[:-4]+'.pdf')
plt.close()


for dataset in datasets:
    rank_path = os.path.join(args.results_path, f'{prefix}{dataset}.pkl')
    aggregate_scores = open_file(rank_path).view(-1)
    _, ranking = torch.sort(aggregate_scores)
    sum_copying = []
    x_axis = []
    for i in range(len(ranking)):
        sum_copying.append((copying_scores[ranking[i:]].sum().item() * 100) / total_copying)
        x_axis.append((100 * i) / len(ranking))
    plt.plot(x_axis, sum_copying, label = DATASET_TO_OFFICIAL[dataset], color=DATASET_TO_COLOR[dataset])


plt.xlabel('Percentage pruned (%)', fontsize=20)
plt.ylabel("% of Total Copying Score Retained", fontsize=20)
plt.title(f'Impact of Pruning Attention Heads on Copying ({args.shot})', wrap=True)
plt.legend()
plt.grid()
os.makedirs(os.path.dirname(args.save_copying_plot_path), exist_ok = True)
plt.savefig(args.save_copying_plot_path)
plt.savefig(args.save_copying_plot_path[:-4]+'.pdf')
plt.close()