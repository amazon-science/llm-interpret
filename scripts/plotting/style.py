import matplotlib.pyplot as plt 
import seaborn as sns

params = {
    "axes.titlesize": 22,
    "legend.fontsize": 16,
    "figure.figsize": (12, 8),
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.titlesize": 22,
    "font.family": "Liberation Mono"
}

plt.rcParams.update(params)
plt.style.use("seaborn-whitegrid")
sns.set_style("white")

DATASET_TO_OFFICIAL = {'hellaswag': 'HellaSwag',
                        'piqa': 'PIQA',
                        'arc_easy': 'ARC (Easy)',
                        'arc_challenge': 'ARC (Challenge)',
                        'openbookqa': 'OpenBookQA',
                        'winogrande': 'Winogrande',
                        'boolq': 'BoolQ',
                        'cb': 'CB',
                        'wic': 'WIC',
                        'wsc': 'WSC',
                        'multirc': 'MultiRC',
                        'rte': 'RTE',
                        'record': 'ReCoRD',
                        'copa': 'COPA',
                        'lambada': 'LAMBADA',
                        'mathqa': 'MathQA'
                        }

DATASET_TO_MARKER = {'hellaswag': 'o',
                        'piqa': 'o',
                        'arc_easy': 'o',
                        'arc_challenge': 'o',
                        'openbookqa': 'o',
                        'winogrande': 'o',
                        'boolq': 'o',
                        'cb': 'o',
                        'wic': '^',
                        'wsc': '^',
                        'multirc': '^',
                        'rte': '^',
                        'record': '^',
                        'copa': '^',
                        'lambada': '^',
                        'mathqa': '^'
                        }

DATASET_TO_COLOR = {'hellaswag': 'maroon',
                        'piqa': 'chocolate',
                        'arc_easy': 'greenyellow',
                        'arc_challenge': 'violet',
                        'openbookqa': 'royalblue',
                        'winogrande': 'crimson',
                        'boolq': 'slategrey',
                        'cb': 'darkkhaki',
                        'wic': 'orangered',
                        'wsc': 'gold',
                        'multirc': 'lime',
                        'rte': 'steelblue',
                        'record': 'cyan',
                        'copa': 'magenta',
                        'lambada': 'saddlebrown',
                        'mathqa': 'gray'
                        }