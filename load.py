import os
import numpy as np

from typing import List, Tuple, TYPE_CHECKING, Dict, Set

if TYPE_CHECKING:
    from loader import Loader
    from word_problem import WordProblem

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

enc = 'utf8'


def count_in_dir(directory: str, wp_files_count: int = 4) -> int:
    """
    Load count of word problems i directory. Can differ. For use change divider.
    IMPORTANT: When loading word problems from './data/traindata' wp_files_count needs to be set to 3.
    :param wp_files_count: Integer specifying count of files for one word problem, default is 4 (wp, result).
    :param directory: Goal directory.
    :return: Count of word problems.
    """
    return int(
        len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]) / wp_files_count)


def get_files_for_svm(directory: str, loader: 'Loader', count: int = None) -> Tuple:
    """
    Gets files for svm and expressions processing.
    Use example:
    get_files_for_svm('./dataset/WP500/traindata', loader)
    :param count: Specifying the number of files.
    :param loader: Instance of class Loader.
    :param directory: String. Specified path.
    :return: Dictionary of numpy arrays for word_problems, results and expressions.
    """
    from word_problem import WordProblem
    files = {'sentences', 'results', 'expressions'}
    dictionary = get_specified_files(directory, files, count)
    word_problems = [WordProblem(sentence, loader) for sentence in dictionary['sentences']]
    return np.array(word_problems), np.array(dictionary['results']), np.array(dictionary['expressions'])


def get_files_for_sa(directory: str) -> Tuple:
    """
    Gets files for sa and expressions processing.
    Use example:
    get_files_for_svm('./dataset/WP500/traindata', loader)
    :param directory: String. Specified path.
    :return: Dictionary of numpy arrays for word_problems, results and expressions.
    """
    sentences, results, data, expressions = get_all_files(directory)
    return np.array(sentences), np.array(results), np.array(data), np.array(expressions)


def get_specified_files(directory: str, specifications: Set, count: int = None) -> Dict:
    """
    Loads specified files like sentences, data (CONLLU), results and expressions from specified directory.
    Use example:
    get_specified_files('./dataset/WP500/traindata', {'sentences', 'data'})
    :param count: Specifying the number of files.
    :param directory: String. Specifying path where are files saved.
    :param specifications: Set of strings can be: sentences, data, results, expressions.
    :return: Dictionary where key is specification and value is loaded data.
    """
    dictionary = {'sentences': load_sentence_dir, 'data': load_conllu_dir, 'results': load_result_dir,
                  'expressions': load_expression_dir}
    count = count_in_dir(directory) if count is None else count
    assert count != 0
    ret = {specification: list(map(lambda i: dictionary[specification](directory, i), range(1, count + 1))) for
           specification in specifications}
    return ret


def get_all_files(directory: str) -> Tuple:
    """
    Loads all files for word problem and returns them in the array.
    :param directory: String. Specifies directory where are files stored.
    :return: List of String sentences. List of results.
    """
    count = count_in_dir(directory)
    assert count != 0
    sentences, results, data, expressions = [], [], [], []
    for i in range(1, count + 1):
        sentences.append(load_sentence_dir(directory, i))
        results.append(load_result_dir(directory, i))
        data.append(load_conllu_dir(directory, i))
        expressions.append(load_expression_dir(directory, i))
    return sentences, results, data, expressions


def get_files(directory: str, i: int) -> Tuple:
    """
    Loads all 3 files representing word problem.
    :param directory: String. Specified directory.
    :param i: Integer. The number of word problem.
    :return: Sentence, result and data in CONLLU format.
    """
    data = load_conllu_dir(directory, i)
    result = load_result_dir(directory, i)
    sentence = load_sentence_dir(directory, i)
    return np.array(sentence), np.array(result), np.array(data)


def load_raw_txt(path: str) -> str:
    """
    Helper function for loading files.
    :param path: Path to file.
    :return: String loaded data from file.
    """
    with open(path, encoding=enc) as f:
        data = f.read()
    return data


def load_conllu_dir(path: str, file_number: int) -> str:
    """
    Loads data from conllu file specified by path and file_number.
    :param path: String path to files.
    :param file_number: Integer specifying file number.
    :return: String conllu data.
    """
    f = open(path + "/word_problem" + str(file_number) + "conllu.conllu", encoding=enc)
    data = f.read()
    f.close()
    return data


def load_sentence_dir(path: str, file_number: int) -> str:
    """
    Loads data from sentence file specified by path and file_number.
    :param path: String path to files.
    :param file_number: Integer specifying file number.
    :return: String sentence.
    """
    filename = path + "/word_problem" + str(file_number) + ".txt"
    return load_raw_txt(filename)


def load_result_dir(path: str, file_number: int) -> str:
    """
    Loads data from result file specified by path and file_number.
    :param path: String path to files.
    :param file_number: Integer specifying file number.
    :return: String result.
    """
    filename = path + "/word_problem" + str(file_number) + "result.txt"
    data = load_raw_txt(filename).split('\n')[0]
    return data


def load_expression_dir(path: str, file_number: int) -> str:
    """
    Loads data from expression file specified by path and file_number.
    :param path: String path to files.
    :param file_number: Integer specifying file number.
    :return: String expression.
    """
    filename = path + "/word_problem" + str(file_number) + "expression.txt"
    data = load_raw_txt(filename).split('\n')[0]
    return data


def load_sequences_sa() -> List:
    """
    Loads sequences for SA solver
    :return: List of list.
    """
    filename = './data/sequences_sa/sequences.txt'
    data = load_raw_txt(filename).split('\n')
    result = [x.split(" ") for x in data]
    return result


def load_sequences_svm(part: str) -> List:
    """
    Loads sequences for given part.
    :param part: QUESTION or WP flag.
    :return: Loaded data from sequences that are split.
    """
    assert part == 'QUESTION' or part == 'WP'
    with open('./database/sequences_svm/' + part.lower() + '/sequences.txt', 'r', encoding='utf8') as f:
        data = [i.split(' | ') for i in f.read().split('\n')]
    return data


def load_databases() -> Tuple:
    """
    Loads databases of words for the first solution.
    :return: Tuple of databases.
    """
    plus = load_raw_txt('./database/database_sa/plus.txt').split('\n')
    minus = load_raw_txt('./database/database_sa/minus.txt').split('\n')
    return plus, minus


def load_pickle(path: str) -> Dict:
    """
    Loads pickle file from given path.
    :param path: Given path to pickle file.
    :return: Loaded dictionary.
    """
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data


def load_wc_lemma_dict(path: str = './database/wc_lemma_dictionary.p') -> Dict:
    """
    Loads dictionary of word classes and lemmas.
    :param path: Default path.
    :return: Dictionary of where KEY is word and VALUE is tuple word class and lemma.
    """
    return load_pickle(path)


def load_important_wc(path: str = './database/important_wc.p') -> Dict:
    """
    Loads dictionary of important word classes. Important word classes are: VERB', 'PREP', 'CONJ', 'ADV', 'PRON'.
    This file was created by function handle_important_other
    :return: Dictionary of important word classes.
    """
    return load_pickle(path)


def get_results_list(directory: str) -> List:
    """
    Loads all results in directory and stores them in the list.
    :param directory: String. Path where are files saved.
    :return: List of results.
    """
    count = count_in_dir(directory)
    results = [load_result_dir(directory, i) for i in range(1, count + 1)]
    return results


def get_expressions_list(directory: str) -> List:
    """
    Loads all expressions in directory and stores them in the list.
    :param directory: String. Path where are files saved.
    :return: List of expressions.
    """
    count = count_in_dir(directory)
    expressions = [load_expression_dir(directory, i) for i in range(1, count + 1)]
    return expressions


def prepare_word_problems(directory: str, loader: 'Loader') -> List['WordProblem']:
    """
    Creates array of WordProblem objects.
    :param directory: Directory where to pick word problems.
    :param loader: Instance of class Loader.
    :return: List of WordProblem objects.
    """
    from word_problem import WordProblem
    count = count_in_dir(directory)
    sentences = list(map(lambda x: load_sentence_dir(directory, x), range(1, count + 1)))
    word_problems = list(map(lambda x: WordProblem(x, loader), sentences))
    return word_problems


def load_prepared_testdata(loader: 'Loader', path: str) -> List['WordProblem']:
    """
    Loads and prepares array of WordProblem objects of testdata.
    :param path: Specifies path to dataset will be used - './dataset/WP500/testdata'.
    :param loader: Instance of class Loader.
    :return: List of WordProblem objects.
    """
    assert path == './dataset/WP500/testdata' or path == './dataset/WP150/testdata'
    return prepare_word_problems(path, loader)


def load_prepared_traindata(loader: 'Loader', path: str) -> List['WordProblem']:
    """
    Loads and prepares array of WordProblem objects of traindata.
    :param path: Specifies path to dataset will be used - './dataset/WP500/traindata'.
    :param loader: Instance of class Loader.
    :return: List of WordProblem objects.
    """
    assert path == './dataset/WP500/traindata' or path == './dataset/WP150/traindata'
    return prepare_word_problems(path, loader)
