import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mpl_toolkits.mplot3d import Axes3D

from typing import List


def show_points(points: np.ndarray, y: np.ndarray):
    """
    Handler for showing points in 2D or 3D.
    :param points: Points - [x, y] or [x, y, z].
    :param y: Label for every point.
    """
    if points.shape[1] == 1:
        show2d(np.array([np.append(x, [0]) for x in points]), y)
    elif points.shape[1] == 2:
        show2d(points, y)
    elif points.shape[1] == 3:
        show3d(points, y)
    else:
        print('Cannot show.')


def show2d(points: np.ndarray, y: np.ndarray):
    """
    Plots 2D points.
    :param points: [[x1, y1], [x2, y2]]
    :param y: Label for every point.
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(points[:, 0], points[:, 1], c=np.array(y))
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)
    ax.legend()
    plt.show()


def show3d(points: np.ndarray, y: np.ndarray):
    """
    Plots 3D points.
    :param points: [[x1, y1, z1], [x2, y2, z2]]
    :param y: Label for every point.
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xs=points[:, 0], ys=points[:, 1], zs=points[:, 2],
               c=y)
    plt.show()


def bar_graph(expressions: List, accuracy: List, values: List, solver: str, directory: str, language: str = 'en'):
    """
    Plots accuracy bar chart.
    Inspiration from: https://medium.com/python-pandemonium/data-visualization-in-python-bar-graph-in-matplotlib-f1738602e9c4
    :param language: Language of labels.
    :param expressions: List of expressions.
    :param accuracy: List of accuracy of each expression.
    :param values: List of number of word problems in dataset.
    :param solver: String. Used solver (SA - syntax analysis, ML - machine learning)
    :param directory: String. Directory of dataset - ./dataset/WP500/testdata.
    """
    assert language == 'en' or language == 'cz'
    correct = np.array([accuracy[i] * values[i] for i in range(len(values))])
    wrong = values - correct
    index = np.arange(len(expressions))
    correct_patch = mpatches.Patch(color='green',
                                   label='Correct word problems') if language == 'en' else mpatches.Patch(color='green',
                                                                                                          label='Správné řešení')
    wrong_patch = mpatches.Patch(color='red', label='Wrong word problems') if language == 'en' else mpatches.Patch(
        color='red', label='Špatné řešení')
    plt.legend(handles=[correct_patch, wrong_patch])
    plt.bar(index, correct, color='green')
    plt.bar(index, wrong, bottom=correct, color='red')
    if language == 'en':
        plt.xlabel('Expressions', fontsize=10)
        plt.ylabel('Number of word problems', fontsize=10)
        plt.xticks(index, expressions, fontsize=7, rotation=30)
        plt.title('Accuracy of solver ' + solver + ' on dataset ' + directory.split('/')[2])
    else:
        plt.xlabel('Výrazy', fontsize=10)
        plt.ylabel('Počet slovních úloh', fontsize=10)
        plt.xticks(index, expressions, fontsize=7, rotation=30)
        plt.title('Úspěšnost ' + solver + ' na datasetu ' + directory.split('/')[2])
    return plt


def plot_bar_graph(expressions: List, accuracy: List, values: List, solver: str, directory: str, language: str = 'en'):
    """
    Show bar graph.
    :param language: Language of labels.
    :param expressions: List of expressions.
    :param accuracy: List of accuracy of each expression.
    :param values: List of number of word problems in dataset.
    :param solver: String. Used solver (SA - syntax analysis, ML - machine learning)
    :param directory: String. Directory of dataset - ./dataset/WP500/testdata.
    """
    bar_graph(expressions, accuracy, values, solver, directory, language=language).show()


def save_bar_graph(expressions: List, accuracy: List, values: List, solver: str, directory: str, language: str = 'en', name: str = None):
    """
    Show bar graph.
    :param language: Language of labels.
    :param name: File name of graph if is None the name is created.
    :param expressions: List of expressions.
    :param accuracy: List of accuracy of each expression.
    :param values: List of number of word problems in dataset.
    :param solver: String. Used solver (SA - syntax analysis, ML - machine learning)
    :param directory: String. Directory of dataset - ./dataset/WP500/testdata.
    """
    name = name if name is not None else solver + '_' + directory.split('/')[2] + '_' + directory.split('/')[3]
    bar_graph(expressions, accuracy, values, solver, directory, language=language).savefig(name + '.png', bbox_inches='tight')
