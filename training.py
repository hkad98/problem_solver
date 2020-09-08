from help_functions import run_solution
from loader import Loader


def train(loader, solver, dataset, chosen_test=None, chosen_both=None):
    """
    Runs solver on given dataset. It is testing, which means training on TRAIN_DATA and testing on TRAIN_DATA.
    :param chosen_both: List of chosen word problems for both training and testing set.
    :param chosen_test: List of chosen word problems for both training and testing set.
    :param loader: Instance of class Loader.
    :param solver: String specifying type of solve SA - syntax analysis or SVM - support vector machines.
    :param dataset: WP150 or WP500
    """
    run_solution(loader, solver, dataset, chosen_test=chosen_test, chosen_both=chosen_both)


def run_all_train():
    """
    Runs all train sets of dataset WP150 and dataset WP500.
    Runs on both solvers SA and SVM.
    """
    loader_instance = Loader()
    train(loader_instance, 'SA', 'WP150')
    train(loader_instance, 'SA', 'WP500')
    train(loader_instance, 'SVM', 'WP150')
    train(loader_instance, 'SVM', 'WP500')


if __name__ == '__main__':
    run_all_train()
