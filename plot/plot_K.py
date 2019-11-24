import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
plt.rc('font', family='serif')
plt.rc('font', serif='Times New Roman')

DATASET_NAMES = ['mnist_all_data_0_random_niid',
                 'mnist_all_data_0_niid_unbalanced',
                 'synthetic_alpha0_beta0_niid',
                 'synthetic_alpha1_beta1_niid']
FIG_NAME = {'mnist_all_data_0_random_niid': 'mnist_balanced',
            'mnist_all_data_0_niid_unbalanced': 'mnist_unbalanced',
            'synthetic_alpha0_beta0_niid': 'alpha0_beta0_niid',
            'synthetic_alpha1_beta1_niid': 'alpha1_beta1_niid'}

WORKER_NUM = [5, 10, 30, 50, 70, 100]
RESULTS_FILE = ['fedavg4_logistic__wn{}_tn100_sd0_lr0.1_ep5_bs64_a'.format(wn) for wn in WORKER_NUM]
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = "stix"


DATASET = 'mnist_all_data_0_niid_unbalanced'

for DATASET in DATASET_NAMES:
    result_dict = dict()
    dataset_path = f'../result/{DATASET}'
    files = sorted(os.listdir(dataset_path))
    for file in files:
        for i, rfile in enumerate(RESULTS_FILE):
            if rfile in file:
                with open(os.path.join(dataset_path, file, 'metrics.json'), 'r') as load_f:
                    load_dict = eval(json.load(load_f))
                result_dict[str(WORKER_NUM[i])] = load_dict['loss_on_train_data'][:80]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    COLORS = {"10": colors[0],
              "30": colors[1],
              "50": colors[2],
              "70": colors[3],
              "100": colors[5]}
    LABELS = {"10": r"$K=10$",
              "30": r"$K=30$",
              "50": r"$K=50$",
              "70": r"$K=70$",
              "100": r"$K=100$"}

    plt.figure(figsize=(4, 3))
    for wn, stat in result_dict.items():
        plt.plot(np.arange(len(stat)), np.array(stat), linewidth=1.0, color=COLORS[wn], label=LABELS[wn])

    plt.grid(True)
    # 0: ‘best', 1: ‘upper right', 2: ‘upper left', 3: ‘lower left'
    plt.legend(loc=0, borderaxespad=0., prop={'size': 10})
    plt.xlabel('Round $(T/E)$', fontdict={'size': 10})
    plt.ylabel('Loss', fontdict={'size': 10})
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    before = 15
    a = plt.axes([0.31, 0.6, .3, .3])
    for wn, stat in result_dict.items():
        plt.plot(np.arange(len(stat))[-before:-5], np.array(stat)[-before:-5], linewidth=1.0, color=COLORS[wn], label=LABELS[wn])

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig = plt.gcf()
    fig.savefig(f'{FIG_NAME[DATASET]}_K.pdf')

