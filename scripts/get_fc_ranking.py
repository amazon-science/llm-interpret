import os
import json
import torch
import pickle
import random
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)


args = parser.parse_args()

def get_accuracy(path, task):
    with open(path) as f:
        x = json.load(f)
    return x['results'][f'{task}']['acc'] * 100

datasets = os.listdir(args.results_dir)

for dataset in datasets:
    fc = []
    baseline_result = get_accuracy(os.path.join(args.results_dir, dataset, 'none.txt'), dataset)
    for i in range(64):
        fc_perf = get_accuracy(os.path.join(args.results_dir, dataset, f'fc_{i}.txt'), dataset)
        fc.append(((baseline_result - fc_perf), i))
    ordered = sorted(fc, key = lambda x: x[0])
    order = list(zip(*ordered))[1]
    os.makedirs(args.save_dir, exist_ok = True)
    with open(os.path.join(args.save_dir, f'{dataset}.pkl'), 'wb') as f:
        pickle.dump(order, f)
        print(f'{dataset} saved!')

print('All done')

