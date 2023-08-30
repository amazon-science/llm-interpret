import os
import json
import torch
import argparse
import numpy as np
from style import *

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str, default=None)
parser.add_argument("--title", type=str, default=None)
parser.add_argument("--save_plot_path", type=str, default=None)
parser.add_argument("--shot", type=str, default="0-shot", choices=["0-shot","1-shot","5-shot"])

args = parser.parse_args()

def get_accuracy(path, task):
    with open(path) as f:
        x = json.load(f)
    return x['results'][f'{task}']['acc'] * 100 if task != "record" else x['results'][f'{task}']['em'] * 100
    
# datasets = ["arc_easy", "wsc", "openbookqa", "piqa", "rte", "cb", "hellaswag", "copa", "wic", "arc_challenge", "winogrande"]# "boolq", "multirc"]
# datasets = ["piqa", "arc_challenge", "openbookqa", "winogrande", "cb", "copa", "wic", "wsc", "rte"]
datasets = ['hellaswag','piqa', 'arc_easy', 'arc_challenge', 'openbookqa', 'winogrande', 'boolq', 'cb', 'copa', 'wic', 'wsc', 'multirc', 'rte', 'record']
prefix = "1shot_" if '1-shot' in args.shot else "5shot_" if '5-shot' in args.shot else ""

percents = [10, 20, 30, 40, 50, 60, 70, 80, 90]
li = []

for dataset in datasets:
    files = os.listdir(os.path.join(args.results_path, dataset))
    num_files = list(filter(lambda x: '_percent.txt' in x and len(x)==14, files))
    # num_files = list(filter(lambda x: '_percent.txt' in x and '1shot_' in x, files))
    if len(num_files) >= 9:
    # if len(num_files) == 10:
        print(dataset)
        pruning_nums = []
        baseline_path = os.path.join(args.results_path, dataset, f'{prefix}0_percent.txt')
        baseline_result = get_accuracy(baseline_path, dataset)
        pruning_nums.append(baseline_result)
        for percent in percents:
            result = get_accuracy(os.path.join(args.results_path, dataset, f'{prefix}{percent}_fc_percent.txt'), dataset)
            # result = get_accuracy(os.path.join(args.results_path, dataset, f'1shot_{percent}_percent.txt'), dataset)
            pruning_nums.append(result)
        li.append(pruning_nums)
        plt.plot([0]+percents, pruning_nums, label = DATASET_TO_OFFICIAL[dataset], alpha = 0.6, marker = DATASET_TO_MARKER[dataset], color=DATASET_TO_COLOR[dataset])

average = np.mean(np.array(li), axis = 0)
plt.plot([0]+percents, average, 'k', linewidth = 3, label = 'Average', marker = 'o')
print(average)
plt.ylabel('Accuracy (%)', fontsize = 20)
plt.xlabel('Percentage pruned (%)', fontsize = 20)
plt.xticks(list(range(0,100,10)))
plt.yticks(list(range(0,100,10)))
plt.title(args.title)
plt.legend(bbox_to_anchor=(1.0,0.75))
# plt.axvline(x=70, linestyle = '--')
plt.axvline(x=10, linestyle = '--')
plt.tight_layout()

os.makedirs(os.path.dirname(args.save_plot_path), exist_ok = True)
plt.savefig(args.save_plot_path)
plt.savefig(args.save_plot_path[:-3]+'pdf')