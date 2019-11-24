import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from  mat4py import loadmat
plt.rc('font', family='serif')
plt.rc('font', serif='Times New Roman')
plt.rcParams["mathtext.fontset"] = "stix"

data_mat = './example.mat'
data = loadmat(data_mat)['res']

base_path = os.path.dirname(".")
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fix_data = dict()
decay_data = dict()
for key, value in data.items():
    if 'fix' in key:
        fix_data[key[3:]] = value
    else:
        decay_data[key[5:]] = value

COLORS = {"1": colors[0],
          "5": colors[1],
          "10": colors[2],
          "50": colors[3],
          "100": colors[4]}
LABELS = {'1': r'$E=1$',
          '5': r'$E=5$',
          '10': r'$E=10$',
          '50': r'$E=50$',
          '100': r'$E=100$'}

data_dict = {"fix": fix_data,
             "decay": decay_data}

for name, value in data_dict.items():
    plt.figure(figsize=(4, 3))
    optimal_value = np.inf
    for key, stat in value.items():
        loss = np.array(stat['loss'])/10
        if key == '1':
            optimal_value = min(np.min(loss), optimal_value)
            plt.hlines(optimal_value, 0, len(loss), label='optimal', color=colors[6], linestyles='-.')
            print(optimal_value)

        plt.plot(np.arange(len(loss)), np.array(loss), linewidth=1.0, color=COLORS[key], label=LABELS[key])

    plt.grid(True)
    plt.legend(loc=0, borderaxespad=0., prop={'size': 10})
    plt.ylabel(r'Global loss', fontdict={'size': 10})
    plt.xlabel(r'Round $(T/E)$', fontdict={'size': 10})
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xscale('log')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(f'{name}.pdf')
