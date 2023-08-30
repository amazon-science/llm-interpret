import os
import torch
import pickle
import random
import argparse
import numpy as np
from scipy import stats
from style import *

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str, default=None)
parser.add_argument("--save_plot_path", type=str, default=None)
parser.add_argument("--random", action = "store_true")
parser.add_argument("--aggregate", action = "store_true")
parser.add_argument("--shot", type=str, default="0-shot", choices=["0-shot","1-shot","5-shot"])

args = parser.parse_args()

datasets = ['hellaswag','piqa', 'arc_easy', 'arc_challenge', 'openbookqa', 'winogrande', 'boolq', 'cb', 'copa', 'wic', 'wsc', 'multirc', 'rte', 'record']
prefix = "1shot_" if '1-shot' in args.shot else "5shot_" if '5-shot' in args.shot else ""

# datalengths = [5000, 5000, 635, 5000, 5000, 400, 3668, 5000, 2251, 1119, 250, 5000, 5000, 4957, 2490, 5000, 259, 5000]

rankings = []
aggregate, count = torch.zeros(4608), 0
new_xticks = []
for i in range(len(datasets)):
    pth = os.path.join(args.results_path, f'{prefix}{datasets[i]}.pkl')
    with open(pth, 'rb') as f:
        res = pickle.load(f).view(-1)
        new_xticks.append(f'{DATASET_TO_OFFICIAL[datasets[i]]}')
        print(datasets[i])
        aggregate = aggregate + (res)
        count += 1
    ranking = torch.sort(res)[1].unsqueeze(0)
    rankings.append(ranking)

if args.aggregate:
    aggregate = aggregate / count 
    new_xticks.append('Aggregate')
    rankings.append(torch.sort(aggregate)[1].unsqueeze(0))
    with open(f'logs/head_importance/opt66b/{prefix}aggregate.pkl', 'wb') as f:
        pickle.dump(aggregate, f)
        print('Aggregated Ranking Saved!')

if args.random:
    random_li = list(range(4608))
    random.shuffle(random_li)
    random_li = torch.tensor(random_li)
    new_xticks.append('Random')
    rankings.append(torch.sort(random_li)[1].unsqueeze(0))

rankings = torch.cat(rankings).numpy() # Num_tasks, Dimension
print(rankings.shape)
num_tasks, _ = rankings.shape

matrix = stats.spearmanr(rankings, rankings, axis = 1).correlation
print(matrix.shape)
matrix = matrix[:num_tasks, :num_tasks]

ax = sns.heatmap(matrix, xticklabels = new_xticks, yticklabels = new_xticks, cmap="YlGnBu", annot = True, vmax = 0.5)

plt.title(f'Spearman Rank Correlations between Importance Score Orders')
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 
if args.aggregate:
    ax.hlines([len(datasets)], *ax.get_xlim(), colors = 'C1', linewidth=3)
    ax.vlines([len(datasets)], *ax.get_ylim(), colors = 'C1', linewidth=3)
# ax.figure.tight_layout()
# fig = plt.gcf()
# fig.set_size_inches(20, 12.5)
# fig.set_dpi(100)
plt.tight_layout()

os.makedirs(os.path.dirname(args.save_plot_path), exist_ok = True)
plt.savefig(args.save_plot_path)
plt.savefig(args.save_plot_path[:-3]+'pdf')