import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
plt.rc('font', family='serif')
plt.rc('font', serif='Times New Roman')

DATASET_NAMES = ['mnist_all_data_0_equal_niid',
                 'mnist_all_data_0_niid_unbalanced',
                 'synthetic_alpha0_beta0_niid',
                 'synthetic_alpha1_beta1_niid']

FIG_NAME = {'mnist_all_data_0_equal_niid': 'mnist_balanced',
            'mnist_all_data_0_niid_unbalanced': 'mnist_unbalanced',
            'mnist_all_data_0_random_niid': 'mnist_random',
            'synthetic_alpha0_beta0_niid': 'synthetic0_0',
            'synthetic_alpha1_beta1_niid': 'synthetic1_1'}

# RESULTS_FILE = {'fedavg4_logistic__wn30_tn100_sd0_lr0.1_ep20_bs64_a': 'Scheme I',
#                 'fedavg4_logistic__wn30_tn100_sd0_lr0.1_ep20_bs64_w': 'Scheme II',
#                 'fedavg9_logistic__wn30_tn100_sd0_lr0.1_ep20_bs64_a': 'T-Scheme II',
#                 'fedavg5_logistic__wn30_tn100_sd0_lr0.9_ep20_bs64_a': 'Original'}

RESULTS_FILE = {'fedavg4_logistic__wn10_tn100_sd0_lr0.1_ep10_bs24_a': 'Scheme I',
                'fedavg4_logistic__wn10_tn100_sd0_lr0.1_ep10_bs24_w': 'Scheme II',
                'fedavg9_logistic__wn10_tn100_sd0_lr0.1_ep10_bs24_a': 'T-Scheme II',
                'fedavg5_logistic__wn10_tn100_sd0_lr0.9_ep10_bs24_a': 'Original'}

matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = "stix"


for DATASET in DATASET_NAMES[-2:]:
    result_dict = dict()
    dataset_path = f'../result/{DATASET}'
    files = sorted(os.listdir(dataset_path))
    for file in files:
        for rfile, name in RESULTS_FILE.items():
            if rfile in file:
                with open(os.path.join(dataset_path, file, 'metrics.json'), 'r') as load_f:
                    load_dict = eval(json.load(load_f))
                result_dict[name] = load_dict['loss_on_train_data'][:200]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    COLORS = {"Scheme I": colors[2],
              "Scheme II": colors[1],
              "Original": colors[0],
              'T-Scheme II': colors[3]}

    plt.figure(figsize=(4, 3))
    for name, stat in result_dict.items():
        plt.plot(np.arange(len(stat)), np.array(stat), linewidth=1.0, color=COLORS[name], label=name)

    plt.grid(True)
    # 0: ‘best', 1: ‘upper right', 2: ‘upper left', 3: ‘lower left'

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, loc=0, borderaxespad=0., prop={'size': 10})
    plt.xlabel('Round $(T/E)$', fontdict={'size': 10})
    plt.ylabel('Global loss', fontdict={'size': 10})
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    before = 30
    a = plt.axes([0.31, 0.6, .3, .3])
    for name, stat in result_dict.items():
        plt.plot(np.arange(len(stat))[-before:], np.array(stat)[-before:], linewidth=1.0, color=COLORS[name], label=name)

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig = plt.gcf()
    fig.savefig(f'{FIG_NAME[DATASET]}_scheme.pdf')
