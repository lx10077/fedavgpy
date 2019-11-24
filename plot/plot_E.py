import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
plt.rc('font', family='serif')
plt.rc('font', serif='Times New Roman')
plt.rcParams["mathtext.fontset"] = "stix"


def smooth(y):
    return gaussian_filter1d(y, sigma=0.6)


base_path = os.path.dirname(".")
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

COLORS = {"mnist_balanced": colors[0],
          "mnist_unbalanced": colors[1],
          "synthetic_a0b0": colors[2],
          "synthetic_a1b1": colors[3]}
LABELS = {'mnist_balanced': 'mnist balanced',
          'mnist_unbalanced': 'mnist unbalanced',
          'synthetic_a0b0': r'synthetic$(0,0)$',
          "synthetic_a1b1": r'synthetic$(1,1)$'}
synthetic_a0b0_X = [5, 10, 20, 30, 50, 60, 80, 100, 125, 200]
synthetic_a0b0 = smooth([160, 101, 88, 87, 90, 92, 96, 106, 114, 139])
synthetic_a1b1_X = [5, 10, 20, 30, 50]
synthetic_a1b1 = smooth([189, 140, 143, 150, 194])
mnist_balanced_X = [10, 20, 30, 50, 60, 80, 100, 125, 150]
mnist_balanced = smooth([120, 50, 39, 28, 27, 22, 21, 20, 20])
mnist_unbalanced_X = [10, 20, 30, 50, 100, 200, 400]
mnist_unbalanced = smooth([400, 145, 114, 104, 119, 137, 205])
matplotlib.rcParams['font.family'] = 'Times New Roman'


stats_dict = {'mnist_unbalanced': (mnist_unbalanced_X, mnist_unbalanced),
              'mnist_balanced': (mnist_balanced_X, mnist_balanced),
              'synthetic_a0b0': (synthetic_a0b0_X, synthetic_a0b0),
              'synthetic_a1b1': (synthetic_a1b1_X, synthetic_a1b1)}

plt.figure(figsize=(4, 3))
for data, stat in stats_dict.items():
    plt.plot(np.array(stat[0])*10, np.array(stat[1]), linewidth=1.0, color=COLORS[data], label=LABELS[data])

plt.grid(True)
plt.legend(loc=0, borderaxespad=0., prop={'size': 10})
plt.ylabel(r'Required rounds ($T_{\epsilon}/E$)', fontdict={'size': 10})
plt.xlabel('Local steps ($E$)', fontdict={'size': 10})
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xscale('log')
plt.tight_layout()
fig = plt.gcf()
fig.savefig('E.pdf')
