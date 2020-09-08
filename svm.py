from create_point import workflow
from sklearn import svm
from loader import Loader
from sklearn import preprocessing
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.decomposition import PCA
from sequence_handler import decide

from typing import List, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from data import Data

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


def prepare_traindata_genetic(data: 'Data', loader: 'Loader') -> Tuple:
    """
    Creates points - x and labels - y from the given array.
    :param data: Instance of class Data stores word_problems.
    :param loader: Instance of class Loader.
    :return: x_points represents points and labels represents labels to points
    """
    x_points = [list(workflow(wp, loader).values()) for wp in data.word_problems]
    labels = data.labels
    return x_points, labels


def train_classifier(x_points: List, y: List, kernel: str = 'rbf', parameter_c: int = 10):
    """
    Runs classifier on x_points and labels y. If kernel linear is chosen it runs method LinearSVC
    it is faster method (liblinear implementation).
    :param parameter_c: Integer value representing C parameter for SVM.
    :param kernel: Kernel which will be used for SVM algorithm.
    :param x_points: Normalized points of word_problem.
    :param y: Labels for word problems.
    :return: Classifier.
    """
    assert kernel == 'rbf' or kernel == 'linear' or kernel == 'poly'
    clf = svm.LinearSVC(C=parameter_c, max_iter=10000) if kernel == 'linear' else svm.SVC(C=parameter_c,
                                                                                          kernel=kernel)
    clf.fit(x_points, y)
    return clf


def test_genetic(classifier, x_test_points: np.ndarray, x_test_real: np.ndarray) -> float:
    """
    Tests genetic data.
    :param x_test_real: Real labels for points.
    :param x_test_points: Points representing word problem. They need to be scale_on_train.
    :param classifier: Classifier created by svm.SVM
    :return: Accuracy of computation.
    """
    predictions = classifier.predict(x_test_points)
    real = np.array(x_test_real)
    same = predictions == real
    return np.sum(same) / predictions.size


def train_genetic(loader: 'Loader', data: 'Data', kernel: str = 'rbf', parameters_c: List = None) -> Tuple:
    """
    Applies SVM algorithm to word problems from genetic.py.
    :param parameters_c: List of C values to be tested.
    :param kernel: Kernel which will be used for SVM algorithm.
    :param data: Instance of class data.
    :param loader: Instance of Loader class.
    :return: Classifier, x_points, y.
    """
    if parameters_c is None:
        parameters_c = [10]
    x_points, y = prepare_traindata_genetic(data, loader)
    x_points = preprocessing.scale(x_points)
    classifiers = [train_classifier(x_points, y, kernel=kernel, parameter_c=c) for c in
                   parameters_c] if parameters_c is not None else [
        train_classifier(x_points, y, kernel=kernel, parameter_c=10)]
    return classifiers, x_points, y


def train(x_train_points: List, labels: List, preprocess: bool = True):
    """
    Applies SVM algorithm to word problems represented as points in dimension n, where n is the number of unique expressions in training set.
    :param labels: List of values representing classes of points.
    :param x_train_points: List of points
    :param preprocess: Boolean if the preprocess of points is needed.
    :return: Returns classifier.
    """
    preprocessed_train = preprocessing.scale(x_train_points) if preprocess else x_train_points
    return train_classifier(preprocessed_train, labels)


def scale_on_train(x_train_points: List, x_test_points: List) -> List:
    """
    Scales each point in x_test_points on x_train_points.
    :param x_train_points: Not scaled x_train_points.
    :param x_test_points: Not scaled x_test_points.
    """
    preprocessed_test = []
    for test_point in x_test_points:
        x_train_points.append(test_point)
        preprocessed = preprocessing.scale(x_train_points)
        x_train_points = x_train_points[:-1]
        preprocessed_test.append(preprocessed[-1])
    return preprocessed_test


def find_sequence_handler(loader: 'Loader', predictions: List, word_problems_test: List):
    """
    If find_sequences flag is True, it performs finding sequence.
    :param loader: Instance of class Loader.
    :param predictions: List of predictions.
    :param word_problems_test: List of WordProblem instances.
    """
    for i, w in enumerate(word_problems_test, start=0):
        flag, expression = decide(loader, w)
        if flag:
            predictions[i] = eval(expression, {}, word_problems_test[i].num_dict_wp)


def my_svm_given(loader: 'Loader', train_data: 'Data', test_data: 'Data', find_sequences: bool = True) -> np.ndarray:
    """
    Function that performs SVM on given train_data and test_data.
    :param find_sequences: Boolean tag representing if sequence will be found.
    :param test_data: Instance of class Data, representing test data wrapper.
    :param train_data: Instance of class Data, representing train data wrapper.
    :param loader: Instance of class Loader.
    :return: Boolean list where True represents correct result, False represents wrong result.
    """
    word_problems_train, train_results = train_data.word_problems, train_data.results
    word_problems_test, test_results = test_data.word_problems, test_data.results

    x_train_points = [list(workflow(wp, loader).values()) for wp in
                      word_problems_train]
    classifier = train(x_train_points, train_data.labels)
    x_test_points = [list(workflow(wp, loader).values()) for wp in
                     word_problems_test]
    preprocessed_test = scale_on_train(x_train_points, x_test_points)
    predictions = classifier.predict(preprocessed_test)
    inv_expressions = {v: k for k, v in test_data.expression_labels.items()}
    predictions = [
        eval(inv_expressions[p], {}, word_problems_test[i].num_dict_wp) if inv_expressions[p] in word_problems_test[
            i].possible_expressions else -1 for i, p in enumerate(predictions)]
    test_results = list(map(lambda x: int(x), test_results))
    if find_sequences:
        find_sequence_handler(loader, predictions, word_problems_test)
    return np.array(predictions) == np.array(test_results)


def highest_value(loader: 'Loader', test_data: 'Data', find_sequences: bool = True) -> np.ndarray:
    """
    This heuristic picks the highest expression - value. Has a lot of common with svm, but the weights need to be ones.
    :param loader: Instance of class Loader.
    :param test_data: Instance of class Data, representing test data wrapper.
    :param find_sequences: Flag representing if it is needed to find sequences.
    :return: Boolean list where True represents correct result, False represents wrong result.
    """
    word_problems_test, test_results = test_data.word_problems, test_data.results
    x_test_points = [list(workflow(wp, loader).values()) for wp in
                     word_problems_test]
    predictions = [point.index(max(point)) for point in x_test_points]
    inv_expressions = {v: k for k, v in test_data.expression_labels.items()}
    predictions = [eval(inv_expressions[p], {}, word_problems_test[i].num_dict_wp) for i, p in enumerate(predictions)]
    test_results = list(map(lambda x: int(x), test_results))
    if find_sequences:
        find_sequence_handler(loader, predictions, word_problems_test)
    return np.array(predictions) == np.array(test_results)


def evaluate_genetic(data: 'Data', loader: Loader, parameters_c: List = None, kernel: str = 'rbf'):
    """
    Special function for genetic algorithm. Uses word problems as train and test.
    :param kernel: String specifying kernel type.
    :param parameters_c: List of C values.
    :param data: Instance of class Data, representing train and test data for genetic algorithm.
    :param loader: Instance of Loader class.
    :return: Accuracy of SVM.
    """
    if parameters_c is None:
        parameters_c = [10]
    classifiers, x_points, y = train_genetic(loader, data, parameters_c=parameters_c, kernel=kernel)
    preprocessed_test_points = scale_on_train(x_train_points=x_points.tolist(), x_test_points=x_points.tolist())
    preprocessed_test_points = np.array(preprocessed_test_points)
    accuracies = [test_genetic(classifier, preprocessed_test_points, y) for classifier in classifiers]
    return accuracies


def lda(data: 'Data', loader: 'Loader') -> np.float:
    """
    Runs Linear Discriminative Analysis on word problems.
    :param data: Instance of class Data.
    :param loader: Instance of class Loader.
    :return: Accuracy of classifier created by LDA.
    """
    x_points = [list(workflow(wp, loader).values()) for wp in data.word_problems]
    x_points = np.array(x_points)
    y = np.array(data.labels)
    classifier = LinearDiscriminantAnalysis()
    classifier.fit(x_points, y).transform(x_points)
    return test_genetic(classifier, x_points, y)


def svm_get_points(data: 'Data', loader: 'Loader') -> List:
    """
    Is useful only if dimension is 2d (the number of expression is 2). Is used for visualization.
    :param data: Instance of class Loader.
    :param loader: Instance of class Loader.
    :return: Points.
    """
    x_points, _ = prepare_traindata_genetic(data, loader)
    return preprocessing.scale(x_points)


def lda_get_points(data: 'Data', loader: 'Loader', n_components: int = 2) -> List:
    """
    Performs Linear discriminant analysis. The number of components is 2. Returns points.
    :param data: Instance of class Data stores word_problems.
    :param loader: Instance of class Loader.
    :param n_components: Number of components, default number is 2. lda_get_points is mainly used for visualization and n_components should be 2 or 3.
    :return: Points.
    """
    x_points, y = prepare_traindata_genetic(data, loader)
    classifier = LinearDiscriminantAnalysis(n_components=n_components)
    return classifier.fit(x_points, y).transform(x_points)


def pca_get_points(data: 'Data', loader: 'Loader', n_components: int = 2) -> List:
    x_points, y = prepare_traindata_genetic(data, loader)
    x_points = np.array(x_points)
    pca = PCA(n_components=n_components)
    pca.fit(x_points)
    return pca.transform(x_points)
