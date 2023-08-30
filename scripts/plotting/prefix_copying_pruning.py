import os
import torch
import pickle
import argparse
import numpy as np
from style import *

parser = argparse.ArgumentParser()
parser.add_argument("--aggregate_path_0_shot", type=str, default=None)
parser.add_argument("--aggregate_path_1_shot", type=str, default=None)
parser.add_argument("--aggregate_path_5_shot", type=str, default=None)
parser.add_argument("--prefix_matching_path", type=str, default=None)
parser.add_argument("--copying_path", type=str, default=None)
parser.add_argument("--save_prefix_plot_path", type=str, default=None)
parser.add_argument("--save_copying_plot_path", type=str, default=None)

args = parser.parse_args()

def open_file(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res

prefix_matching_scores = open_file(args.prefix_matching_path)['mean'].view(-1)
copying_scores = open_file(args.copying_path)['mean'].view(-1)
total_prefix = prefix_matching_scores.sum()
total_copying = copying_scores.sum()

shots = [0, 1, 5]
aggregate_paths = [args.aggregate_path_0_shot, args.aggregate_path_1_shot, args.aggregate_path_5_shot]

for shot, aggregate_path in zip(shots,aggregate_paths):
    aggregate_scores = open_file(aggregate_path).view(-1)
    assert aggregate_scores.shape == prefix_matching_scores.shape == copying_scores.shape
    _, ranking = torch.sort(aggregate_scores)
    sum_prefix = []
    x_axis = []
    for i in range(len(ranking)):
        sum_prefix.append((prefix_matching_scores[ranking[i:]].sum().item() * 100) / total_prefix)
        x_axis.append((100 * i) / len(ranking))
    plt.plot(x_axis, sum_prefix, label = f'{shot}-shot')

plt.xlabel('Percentage pruned (%)', fontsize=20)
plt.ylabel('% of Total Prefix Matching Score Retained', fontsize=20)
plt.title('Impact of Pruning Attention Heads on Prefix Matching', wrap=True)
plt.legend()
plt.grid()
os.makedirs(os.path.dirname(args.save_prefix_plot_path), exist_ok = True)
plt.savefig(args.save_prefix_plot_path)
plt.savefig(args.save_prefix_plot_path[:-4]+'.pdf')
plt.close()


for shot, aggregate_path in zip(shots,aggregate_paths):
    aggregate_scores = open_file(aggregate_path).view(-1)
    assert aggregate_scores.shape == prefix_matching_scores.shape == copying_scores.shape
    _, ranking = torch.sort(aggregate_scores)
    sum_copying = []
    x_axis = []
    for i in range(len(ranking)):
        sum_copying.append((copying_scores[ranking[i:]].sum().item() * 100) / total_copying)
        x_axis.append((100 * i) / len(ranking))
    plt.plot(x_axis, sum_copying, label = f'{shot}-shot')


plt.xlabel('Percentage pruned (%)', fontsize=20)
plt.ylabel("% of Total Copying Score Retained", fontsize=20)
plt.title('Impact of Pruning Attention Heads on Copying', wrap=True)
plt.legend()
plt.grid()
os.makedirs(os.path.dirname(args.save_copying_plot_path), exist_ok = True)
plt.savefig(args.save_copying_plot_path)
plt.savefig(args.save_copying_plot_path[:-4]+'.pdf')
plt.close()