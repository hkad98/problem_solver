import load
import numpy as np
from expression import get_chosen_word_problems
from typing import List, TYPE_CHECKING, Set, Dict

if TYPE_CHECKING:
    from loader import Loader


def prepare_data(loader: 'Loader', chosen: List = None, directory: str = './dataset/WP500/traindata',
                 count=None) -> 'Data':
    """
    Prepares single data for genetic algorithm or solver.
    :param count: Specifies count of word problems in directory.
    :param loader: Instance of class Loader.
    :param chosen: Specification of chosen expressions.
    :param directory: Path to saved word problems.
    :return: Instance of class Data.
    """
    data = Data(directory=directory, loader=loader, chosen=chosen, count=count)
    data.create_expression_labels()
    data.get_result_labels()
    data.set_init_dictionary()
    data.set_possible_expressions_wps(set(data.expressions))
    return data


def create_buckets_new(expressions):
    """
    Creates dictionary where KEY is an expression and VALUE is a list of indexes.
    :param expressions: List of string expressions. ['NUM1 + NUM2', 'NUM1 - NUM2', 'NUM1 + NUM2', ...]
    :return: Dictionary described above. {'NUM1 + NUM2': [0, 2, ....], 'NUM1 - NUM2': [1, ...]}
    """
    buckets = dict()
    for i, expression in enumerate(expressions, start=0):
        if expression in buckets:
            buckets[expression].append(i)
        else:
            buckets[expression] = [i]
    return buckets


def create_expression_labels(expressions):
    """
    Creates sorted dictionary where KEY is an expression and VALUE is an unique number from 0 to N - 1 where N is number of unique expressions.
    :param expressions: List of string expressions. ['NUM1 + NUM2', 'NUM1 - NUM2', 'NUM1 + NUM2', ...]
    :return: {'NUM1 + NUM2': 0, 'NUM1 - NUM2': 1, ...}
    """
    unique = list(set(expressions))
    unique.sort()
    expression_labels = {val: i for i, val in enumerate(unique, start=0)}
    return expression_labels


class Data:
    expression_labels = None
    labels = None

    def __init__(self, directory: str, loader: 'Loader', chosen: List[str] = None, count: int = None) -> None:
        """
        Constructor for class Data. Data class wraps all important information about train or test data.
        :param count: Specifying the number of files for word problems.
        :param directory: Parameter specifying path to files.
        :param loader: Instance of class Loader.
        :param chosen: List of chosen word problems. Example: ['NUM1 + NUM2', 'NUM1 - NUM2', ...]
        """
        self.directory = directory
        if chosen is None:
            self.word_problems, self.results, self.expressions = get_chosen_word_problems(loader, directory,
                                                                                          count=count)
        else:
            self.word_problems, self.results, self.expressions = get_chosen_word_problems(loader, directory, chosen,
                                                                                          count=count)
        self.buckets = create_buckets_new(self.expressions)

    def create_expression_labels(self):
        """
        Creates sorted dictionary where KEY is an expression and VALUE is an unique number from 0 to N - 1 where N is number of unique expressions.
        """
        self.expression_labels = create_expression_labels(self.expressions)

    def set_expression_labels(self, labels: Dict):
        """
        Sets expressions labels.
        :param labels: Dictionary of labels, where KEY is an expression and VALUE is an unique number from 0 to N - 1 where N is number of unique expressions.
        """
        self.expression_labels = labels

    def set_init_dictionary(self):
        """
        Sets init dictionary to every word problem in self.word_problems.
        """
        init = {e: 0 for e in self.expression_labels}
        for word_problem in self.word_problems:
            word_problem.init_dictionary = init

    def get_result_labels(self):
        """
        Converts expressions to number identifier for SVM.
        """
        assert self.expression_labels is not None
        self.labels = [self.expression_labels[expression] for expression in self.expressions]

    def set_possible_expressions_wps(self, expressions: Set):
        """
        Tries all expressions from train expressions. Creates sets of possible expressions (result is higher or equal to 0)
        :param expressions: Expression set from train data.
        """
        for word_problem in self.word_problems:
            word_problem.set_possible_expressions(expressions)


class SolveDataSVM:
    def __init__(self, dataset, loader, train_dir_init='/traindata', test_dir_init='/testdata', chosen_test=None,
                 chosen_both=None):
        """
        Handles creating two wrappers for train and test data. Applies given logic.
        Dataset can be WP150 or WP500.

        :param dataset: WP150 or WP500
        :param loader: Instance of class Loader.
        :param chosen_test: Specifies if we want to run testing on some special expressions. Important condition is that expressions must be in train Data.
        :param chosen_both: If chosen_bot is specified it creates two wrappers with different word problems but same expressions.
        """
        assert dataset == 'WP150' or dataset == 'WP500'
        train_path = './dataset/' + dataset + train_dir_init
        test_path = './dataset/' + dataset + test_dir_init
        if chosen_both is not None:
            self.train_data = Data(train_path, loader, chosen_both)
            self.test_data = Data(test_path, loader, chosen_both)
        else:
            self.train_data = Data(train_path, loader)
            if chosen_test is None:
                # set chosen to set of expressions from self.train_data
                self.test_data = Data(test_path, loader, chosen=list(set(list(self.train_data.expressions))))
            else:
                # check if set chosen_test - set of expressions from self.train_data is empty set,
                assert set(chosen_test) - set(self.train_data.expressions) == set()
                self.test_data = Data(test_path, loader, chosen=chosen_test)
        self.train_data.create_expression_labels()
        self.test_data.set_expression_labels(self.train_data.expression_labels)

        self.train_data.set_init_dictionary()
        self.test_data.set_init_dictionary()

        self.train_data.get_result_labels()
        self.test_data.get_result_labels()

        self.train_data.set_possible_expressions_wps(set(self.train_data.expressions))
        self.test_data.set_possible_expressions_wps(set(self.train_data.expressions))


class SolveDataSA:
    def __init__(self, directory: str, chosen: List = None):
        """
        Prepares data for Syntax Analysis (first solution in Bachelor thesis).
        :param directory: Directory of word problems.
        :param chosen: List of chosen expressions solving word problems.
        """
        sentences, results, data, expressions = load.get_files_for_sa(directory)
        if chosen is None:
            self.sentences, self.results, self.data, self.expressions = sentences, results, data, expressions
        else:
            mask = expressions == chosen[0]
            for c in chosen[1:]:
                mask = np.logical_or(mask, expressions == c)
            self.sentences, self.results, self.data, self.expressions = sentences[mask], results[mask], data[mask], expressions[mask]
