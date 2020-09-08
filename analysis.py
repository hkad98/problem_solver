import load
import sentence_functions as s
from create_templates import get_template
import numpy as np
import copy

from typing import List, TYPE_CHECKING, Dict, Tuple, Set

if TYPE_CHECKING:
    from loader import Loader

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


def longest_sequence(l1: List, l2: List) -> List:
    """
    Finds longest sequence of two lists. Written by dynamic programming.
    :param l1: One list.
    :param l2: Second list.
    :return: List which is longest sequence.
    """
    table = [[0 for _ in range(len(l2))] for _ in range(len(l1))]
    for i in range(len(l1)):
        for j in range(len(l2)):
            if (i == 0 or j == 0) and l1[i] == l2[j]:
                table[i][j] += 1
            elif l1[i] == l2[j]:
                table[i][j] = table[i - 1][j - 1] + 1
    table = np.array(table)
    indexes = list(np.unravel_index(table.argmax(), table.shape))
    result = []
    while True:
        result = [l1[indexes[0]]] + result
        indexes[0] -= 1
        indexes[1] -= 1
        if indexes[0] < 0 or indexes[1] < 0 or table[indexes[0]][indexes[1]] == 0:
            break
    return result


def find_sequences(loader: 'Loader', word_problems: List) -> Tuple:
    """
    Finds sequence in given word problems.
    :param loader: instance of class Loader.
    :param word_problems: List of word problems.
    :return: Can return set of sequences or dictionaries.
    """

    def add_in_dict(dictionary: Dict, part: List):
        """
        Add part into dictionary according to some rules.
        :param dictionary: Dictionary.
        :param part: String.
        """
        seq = s.construct_tokenized(part)
        if seq in dictionary:
            dictionary[seq] += 1
        else:
            dictionary[seq] = 1

    wp_dict = dict()
    q_dict = dict()
    for i, w in enumerate(word_problems):
        wp_part_main = get_template(w.wp_part, loader.wc_lemma_dictionary, wanted_wc={'NOUN'}, tokenized_flag=True,
                                    start='')
        q_part_main = get_template(w.question_part, loader.wc_lemma_dictionary, wanted_wc={'NOUN'}, tokenized_flag=True,
                                   start='')
        for j in range(i + 1, len(word_problems)):
            wp_part = get_template(word_problems[j].wp_part, loader.wc_lemma_dictionary, wanted_wc={'NOUN'},
                                   tokenized_flag=True, start='')
            q_part = get_template(word_problems[j].question_part, loader.wc_lemma_dictionary, wanted_wc={'NOUN'},
                                  tokenized_flag=True, start='')
            wp_part_sequence = longest_sequence(wp_part_main, wp_part)
            q_part_sequence = longest_sequence(q_part_main, q_part)
            if len(wp_part_sequence) > 2:
                add_in_dict(wp_dict, wp_part_sequence)
            if len(q_part_sequence) > 2:
                add_in_dict(q_dict, q_part_sequence)
    return {k for k in wp_dict.keys() if wp_dict[k] > 1}, {k for k in q_dict.keys() if q_dict[k] > 1}
    # return wp_dict, q_dict


def my_merge(dst: Dict, src: Dict):
    """
    Function for merging two dictionaries into one.
    :param dst: Destination dictionary.
    :param src: Source dictionary.
    """
    for key in src:
        if key not in dst:
            dst[key] = {}
        for key2 in src[key]:
            if key2 not in dst[key]:
                dst[key][key2] = 1
            else:
                dst[key][key2] += 1


def remove_single(wc_dict: Dict) -> Dict:
    """
    Removes words which count is equal to 1.
    :param wc_dict: Dictionary from creation_important_wc.
    :return: Updated dictionary.
    """
    cpy = copy.deepcopy(wc_dict)
    for wc in wc_dict:
        for word in wc_dict[wc]:
            if wc_dict[wc][word] == 1:
                del cpy[wc][word]
    return cpy


def remove_unwanted(wc_dict: Dict, unwanted: Set) -> Dict:
    """
    Removes unwanted word classes from dictionary.
    :param wc_dict: Dictionary to be removed from.
    :param unwanted: Unwanted word classes.
    :return: Dictionary without unwanted classes.
    """
    cpy = copy.deepcopy(wc_dict)
    for wc in wc_dict:
        if wc in unwanted:
            del cpy[wc]
    return cpy


def filter_wp(expression: str, structure: np.rec, loader: 'Loader', part: str) -> Dict:
    """
    Filters word problems according to expression. And creates dictionary where KEYS are word classes and values are lemmas.
    :param expression: Expression to be filtered.
    :param structure: Record table created by create_struct.
    :param loader: Instance of class Loader.
    :param part: Part of word problem - QUESTION or WP.
    :return: Dictionary where KEYS are word classes and VALUES are sets of lemmas from word problems.
    """
    mask = structure['expressions'] == expression
    word_problems = structure['word_problems'][mask]
    res = {}
    for word_problem in word_problems:
        if part == 'QUESTION':
            my_merge(res, s.create_bucket_dictionary(s.get_question(word_problem)[0], loader))
        else:
            my_merge(res, s.create_bucket_dictionary(s.wp_without_question(word_problem), loader))
    return res


def remake_dict(src_dict: Dict, part: str, res: Dict = None) -> Dict:
    """
    Creates proper dictionary with proper and needed structure.
    WORD_CLASS:
        LEMMA:
            WORD_PROBLEM_PART:
                EXPRESSION:
                    COUNT
    :param src_dict: Dictionary having not wanted structure
    :param part: WP or QUESTION part of word problem.
    :param res: Optional argument dictionary. If is not passed is created new and returned.
    :return:
    """
    res = res if res is not None else dict()
    for e in src_dict:
        for wc in src_dict[e]:
            for word in src_dict[e][wc]:
                value = src_dict[e][wc][word]
                if wc not in res:
                    res[wc] = {word: {part: {e: value}}}
                else:
                    if word not in res[wc]:
                        res[wc][word] = {part: {e: value}}
                    else:
                        if part not in res[wc][word]:
                            res[wc][word][part] = {e: value}
                        else:
                            res[wc][word][part][e] = value
    return res


def create_important_wc(loader: 'Loader', structure: np.rec, counts: Dict):
    """
    Creates dictionary of important word classes. The dictionary structure is:
    WORD_CLASS:
        LEMMA:
            WORD_PROBLEM_PART:
                EXPRESSION:
                    COUNT
    Example:
    {'PREP': {'na': {'WP': {'NUM1 + NUM1 / NUM2': 4, ...}, 'QUESTION':{...}}, ... }}
    _________________________________________________________________________________
    Use example:
    structure = create_struct(directory, load.count_in_dir(directory, 3))
    counts = dict(Counter(structure['expressions']))
    creation_important_wc(l, structure, counts)
    :param loader: Instance of class loader.
    :param structure: Record table with keys are word_problems, expressions.
    :param counts:
    """
    result_wp = {}
    result_question = {}
    unwanted = {'NOUN', 'ADJ', 'NUM', 'PUNCT', 'INTJ', 'PART'}
    for e in counts.keys():
        result_wp[e] = filter_wp(e, structure, loader, 'WP')
        result_question[e] = filter_wp(e, structure, loader, 'QUESTION')
        result_wp[e] = remove_unwanted(result_wp[e], unwanted)
        result_question[e] = remove_unwanted(result_question[e], unwanted)
        if counts[e] > 1:
            result_wp[e] = remove_single(result_wp[e])
            result_question[e] = remove_single(result_question[e])
    result_dict = remake_dict(result_wp, 'WP')
    remake_dict(result_question, 'QUESTION', result_dict)
    create_pickle('./data/important_wc_newest', result_dict)


def create_pickle(filename: str, d: Dict):
    """
    Creates pickle file.
    :param filename: String of filename.
    :param d: Dictionary to be saved.
    """
    with open(filename + '.p', 'wb') as fp:
        pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)


def create_struct(directory: str, count: int = None) -> np.rec:
    """
    Creates record table from 2 np arrays. Sentences and expressions loaded from directory.
    :param count: Specifying the number of files.
    :param directory: Path to files.
    :return:
    """
    files_data = load.get_specified_files(directory, {'sentences', 'expressions'}, count)
    return np.rec.fromarrays((np.array(files_data['sentences']), np.array(files_data['expressions'])),
                             names=('word_problems', 'expressions'))

# INFO: Use example.
# if __name__ == '__main__':
#     from collections import Counter
#     from loader import Loader
#     loader_instance = Loader()
#     train_directory = './data/traindata'
#     my_structure = create_struct(train_directory, load.count_in_dir(train_directory, 3))
#     my_counts = dict(Counter(my_structure['expressions']))
#     create_important_wc(loader_instance, my_structure, my_counts)
