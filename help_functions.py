import load
import os
import sentence_functions as s
import syntax_analysis as sa
import numpy as np
from prettytable import PrettyTable
from visualize import plot_bar_graph, save_bar_graph
from svm import my_svm_given as get_svm_results
from data import SolveDataSVM, SolveDataSA
from collections import OrderedDict


from typing import TYPE_CHECKING, Tuple, List

if TYPE_CHECKING:
    from loader import Loader
    from data import Data


def database_word_handler(question: List) -> Tuple:
    """
    Loads databases and checks number counts how many words from sentence are in database minus a database plus.
    :param question: String. Word problem question.
    :return: Boolean. If rule True else False.
    """
    plus, minus = load.load_databases()
    words = s.get_word_lemma(question)
    result_minus = len([word for word in words if word in minus])
    result_plus = len([word for word in words if word in plus])
    return (result_minus - result_plus) > 0, (result_plus - result_minus) > 0


def contains_other_new(sentences: List) -> bool:
    """
    Simplification of method contains_other. I encourage advanced users to use this method instead of contains_other.
    The reason why this method is not primarily used is because it should cover less cases than method contains_other.
    :param sentences: Sentences in conllu.
    :return:
    """
    no_question_lemmas = [[word[2] for word in sentence if len(word) > 3] for sentence in sentences[0:len(sentences)-1]]
    for sentence in no_question_lemmas:
        if any(np.array(sentence) == 'ostatní'):
            return True
    return False


def contains_other(sentences: List) -> bool:
    """
    This method finds if sentences (without question contains word 'ostatní').
    Checks if some sentence contains "ostatní".
    :param sentences: Sentences in conllu.
    :return: True, False.
    """
    question = [word[2] for word in sentences[-1] if len(word) > 2 and word[1] != '?']
    commas = [[word for word in sentence if
               len(word) > 2] for sentence in sentences if
              len([word for word in sentence if
                   len(word) > 2 and word[1] == 'ostatní']) > 0]
    if len(commas) > 0:
        idx = [word[1] for word in commas[0] if len(word) > 2].index('ostatní')
        after_other = [word[2] for word in commas[0][idx + 1:len(commas[0]) - 1]]
        ok = 0
        for val in after_other:
            if val in question:
                ok += 1
        return True if ok >= len(after_other) / 2 else False
    else:
        sentences = [sentence for sentence in sentences if s.contains_symbol(sentence, 'Ostatní')]
        if len(sentences) > 0:
            after_other = [word[2] for word in sentences[0][1:len(sentences[0]) - 1]]
            ok = 0
            for val in after_other:
                if val in question:
                    ok += 1
            return True if ok >= len(after_other) / 2 else False
        else:
            return False


def clear_directory(directory: str, delete_dir: bool = False) -> None:
    """
    Removes all files in directory and if delete_dir True also removes directory.
    :param directory: Name of directory.
    :param delete_dir: Flag for deleting directory.
    """
    file_list = [f for f in os.listdir(directory)]
    for f in file_list:
        os.remove(os.path.join(directory, f))
    if delete_dir:
        os.rmdir(directory)


def get_sa_results(solve_data: SolveDataSA, loader: 'Loader') -> np.ndarray:
    """
    Runs solver - syntax layer on word problems in directory.
    :param loader: Instance of class Loader.
    :param solve_data: Class that stores data for SA algorithm.
    :return: Numpy array of boolean values, representing if the word problem was solved successfully.
    """
    results = []
    for i, d in enumerate(solve_data.data):
        prediction, flag = sa.solve(d, loader)
        if int(solve_data.results[i]) == int(prediction):
            results.append(True)
        else:
            results.append(False)
    return np.array(results)


def prepare_data_visualize(data: 'Data', results: np.ndarray) -> object:
    """
    Prepares data for visualization.
    :param data: Instance of class Data, representing test data wrapper.
    :param results: Numpy array of boolean values.
    :return: Tuple of numpy arrays - expressions, accuracy, values.
    """
    buckets = OrderedDict(sorted(data.buckets.items()))
    expressions = np.array(list(buckets.keys()))
    values = np.array([len(v) for v in list(buckets.values())])
    accuracy = np.array([np.sum(results[buckets[b]]) / len(buckets[b]) for b in buckets])
    return expressions, accuracy, values


def print_result(data: 'Data', results: np.ndarray, solver: str) -> None:
    """
    Prints table where are 3 columns - Expression, Accuracy, Number of word problems.
    :param data: Instance of class Data, representing test data wrapper.
    :param solver: String. Specifying used solver.
    :param results: Numpy array of boolean values.
    """
    buckets = data.buckets
    t = PrettyTable(['Expression', 'Accuracy', 'Number of word problems'])
    for b in buckets:
        bucket_sum = np.sum(results[buckets[b]])
        t.add_row([b, str(bucket_sum / len(buckets[b])), len(buckets[b])])
    print(t)
    print("Success for dataset " + data.directory.split('/')[2] + " by solver " + solver + ": " + str(
        np.sum(results) / len(results)))


def run_solution(loader: 'Loader', solver: str, dataset: str, train_flag: bool = True, chosen_test: List = None,
                 chosen_both: List = None):
    """
    Handles running testing on test set and train set.
    :param loader: Instance of class Loader.
    :param solver: 'SA' or 'SVM'
    :param dataset: 'WP500' or 'WP150'
    :param train_flag: Optional boolean if run training a testing on train set.
    :param chosen_test: Chosen expressions for testing.
    :param chosen_both: Chosen expressions for training and testing.
    """
    assert solver == 'SA' or solver == 'SVM'
    assert dataset == 'WP500' or dataset == 'WP150'
    test_directory = './dataset/WP500/testdata' if dataset == 'WP500' else './dataset/WP150/testdata'
    train_directory = './dataset/WP500/traindata' if dataset == 'WP500' else './dataset/WP150/traindata'

    if train_flag:
        solve_data = SolveDataSVM(dataset, loader, test_dir_init='/traindata', chosen_test=chosen_test,
                                  chosen_both=chosen_both)
        directory = train_directory
    else:
        solve_data = SolveDataSVM(dataset, loader, chosen_test=chosen_test, chosen_both=chosen_both)
        directory = test_directory

    train_data = solve_data.train_data
    test_data = solve_data.test_data

    if solver == 'SA':
        solve_data = SolveDataSA(train_directory) if train_flag else SolveDataSA(test_directory, chosen=list(
            set(list(train_data.expressions))))
        results_array = get_sa_results(solve_data, loader)
    else:
        results_array = get_svm_results(loader=loader, train_data=train_data,
                                        test_data=test_data, find_sequences=True)
    print_result(test_data, results_array, solver)
    expressions, accuracy, values = prepare_data_visualize(test_data, results_array)
    plot_bar_graph(expressions, accuracy, values, solver, directory, language='en')
    # save_bar_graph(expressions, accuracy, values, solver, directory, language='cz')
