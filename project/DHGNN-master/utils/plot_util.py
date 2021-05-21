import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os.path as osp

font = {'weight': 'bold', 'size': 16}
label_font = {'weight': 'bold', 'size': 20}
axes = {'linewidth': 2}
lines = {'linewidth': 2}

matplotlib.rc('font', **font)           # set font
matplotlib.rc('axes', **axes)           # set axes
matplotlib.rc('lines', **lines)         # set lines

x_k = np.array([4, 8, 16, 32, 64, 128])
y_all = np.array((.814, .811, .812, .816, .820, .825))
y_dhg = np.array((.786, .793, .801, .801, .802, .805))
y_hgc = np.array((.806, .806, .809, .809, .811, .813))


def plot_figure(xs, ys, legends, y_ticks, x_label, filename):
    """
    plot multi-line
    :param xs: list of x data
    :param ys: list of y data
    :param legends: list of legends
    :param y_ticks: y axis ticks
    :param x_label: label string of x axis (y_label: Accuracy)
    :param filename: saved filename
    :return:
    """
    assert(len(xs) == len(ys))                      # length of xs must equal that of ys
    MARKERS = ['o', 's', '^', 'v', 'D']            # iteratively used markers
    LEN_MARKERS = len(MARKERS)

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=.2, right=.9, top=.9, bottom=.2)
    x_ticks = [2, 4, 8, 16, 32, 64, 128, 256]
    for i, x in enumerate(xs):
        ax.semilogx(x, ys[i], marker=MARKERS[i % LEN_MARKERS], basex=2)
    ax.grid()
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xlabel(x_label, fontdict=label_font)
    ax.set_ylabel('Accuracy', fontdict=label_font)
    if len(xs) > 1:
        ax.legend(legends)
    fig.savefig(osp.join('../figures/', filename))


if __name__ == '__main__':
    plot_figure([x_k, x_k], [y_all, y_dhg], ['complete model', 'remove HGC'], np.arange(.78, .84, .01), 'k', 'remove_hgc.png')
    plot_figure([x_k, x_k], [y_all, y_hgc], ['complete model', 'remove DHG'], np.arange(.78, .84, .01), 'k', 'remove_dhg.png')
