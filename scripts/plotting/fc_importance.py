import os
import sys
import json
import torch
import pickle
import argparse
import numpy as np
from style import *

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str, default=None)
parser.add_argument("--base_results_path", type=str, default=None)
parser.add_argument("--save_plot_path", type=str, default=None)
parser.add_argument("--dump_fc_importance", action='store_true')
parser.add_argument("--dump_fc_importance_path", type=str, default=None)
parser.add_argument("--shot", type=str, default="0-shot", choices=["0-shot","1-shot","5-shot"])

args = parser.parse_args()

def get_accuracy(path, dataset):
    with open(path) as f:
        x = json.load(f)
    return x['results'][f'{dataset}']['acc'] * 100 if dataset != "record" else x['results'][f'{dataset}']['em'] * 100
    
# datasets = ['hellaswag','piqa', 'arc_easy', 'arc_challenge', 'openbookqa', 'winogrande', 'boolq', 'cb', 'copa', 'wic', 'wsc', 'multirc', 'rte', 'record']
datasets = ['wic', 'multirc', 'record']
li_avg = []
c = 0
prefix = "1shot_" if '1-shot' in args.shot else "5shot_" if '5-shot' in args.shot else ""
for dataset in datasets:
    if args.base_results_path == None:
        args.base_results_path = args.results_path
    baseline_path = os.path.join(args.base_results_path, dataset, f'{prefix}0_percent.txt')
    baseline_result = get_accuracy(baseline_path, dataset)
    results_collector = []
    for i in range(64):
        fc_score = get_accuracy(os.path.join(args.results_path, dataset, f'{prefix}fc_{i}.txt'), dataset)
        results_collector.append(baseline_result - fc_score)
    print(dataset)
    if args.dump_fc_importance:
        os.makedirs(os.path.dirname(args.dump_fc_importance_path), exist_ok = True)
        with open(f'{args.dump_fc_importance_path}{prefix}{dataset}.pkl', 'wb') as f:
            zipped = list(zip(list(range(64)), results_collector))
            temp = sorted(zipped, key = lambda x: x[1])
            fc_knocking_importance = list(list(zip(*temp))[0])
            print(fc_knocking_importance)
            pickle.dump(fc_knocking_importance, f)
            print(f"Dumped {dataset}")

    li_avg.append(results_collector)
    plt.plot(results_collector, alpha = 0.5, label = DATASET_TO_OFFICIAL[dataset], color=DATASET_TO_COLOR[dataset])

if args.dump_fc_importance:
    sys.exit()

average = np.mean(np.array(li_avg), axis = 0)
plt.plot(average, linewidth=4, label = 'Average', color = 'k')
plt.xticks(list(range(1,65,3)))
# fig = plt.gcf()
# fig.set_size_inches(25, 15)

plt.xlabel('Layer Number', fontsize = 15)
plt.ylabel(f'Accuracy Difference (%)', fontsize = 15)
plt.title(f'FFN Oracle Pruning ({args.shot})', fontsize = 15)
plt.legend(bbox_to_anchor=[1.0, 0.75])
plt.tight_layout()

os.makedirs(os.path.dirname(args.save_plot_path), exist_ok = True)

plt.savefig(args.save_plot_path)
plt.savefig(args.save_plot_path.replace('png', 'pdf'))
plt.close()