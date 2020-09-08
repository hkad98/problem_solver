from create_templates import get_template
from sentence_functions import wp_without_question, \
    sentences_containing_number

from typing import List, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from loader import Loader
    from word_problem import WordProblem


def list_in(a: List, b: List) -> bool:
    """
    Checks if sequence of values from list a are in list b.
    https://stackoverflow.com/questions/51053402/finding-a-sequence-in-list-using-another-list-in-python
    :param a: Sequence of values in list.
    :param b: Destination list.
    :return: Boolean if sequence is in list.
    """
    return any(map(lambda x: b[x:x + len(a)] == a, range(len(b) - len(a) + 1)))


def list_in_index(a: List, b: List) -> List:
    """
    Finds indexes of sequence in a list.
    https://stackoverflow.com/questions/10459493/find-indexes-of-sequence-in-list-in-python
    :param a: Sequence of values in list.
    :param b: Destination list.
    :return: List of tuple indexes.
    """
    return [(i, i + len(a)) for i in range(len(b)) if b[i:i + len(a)] == a]


def sequence_finder(part: List, sequences: List) -> Tuple:
    """
    Tries to find all sequences. Stops at the first one.
    :param part: List of words/template values got by method get_template.
    :param sequences: Sequence for math operation from file.
    :return: Number of match cases.
    """
    ret = ''
    found = False
    for s in sequences:
        seq = s[0].split(' ')
        expr = s[1]
        if '_' in seq:
            idx = seq.index('_')
            l1 = seq[:idx]
            l2 = seq[idx + 1:]
            if list_in(l1, part) and list_in(l2, part):
                ret = expr
                found = True
                break
        else:
            if list_in(seq, part):
                ret = expr
                found = True
                break
    return found, ret


def sequence_handler_part(loader: 'Loader', part: str, part_tag: str) -> Tuple:
    """
    Finds sequences in part of word problem - QUESTION or WP.
    :param loader: Instance of class Loader.
    :param part: Part of word problem.
    :param part_tag: Flag defining which part will be handled.
    :return:
    """
    assert part_tag == 'WP' or part_tag == 'QUESTION'
    wp_template = get_template(part, loader.wc_lemma_dictionary, start='', tokenized_flag=True, wanted_wc={'NOUN'})
    sequences = loader.wp_sequences_svm if part_tag == 'WP' else loader.question_sequences_svm
    return sequence_finder(list(wp_template), sequences)


def sequence_handler(loader: 'Loader', word_problem: 'WordProblem') -> Tuple:
    """
    Finds sequences in word problem.
    :param loader: Instance of class Loader.
    :param word_problem: Instance of class WorProblem.
    :return: Returns 4 values: (flag for part WP and if flag is True it returns sequence else []) and (flag for part QUESTION and if flag is True it returns sequence else [])
    """
    wp_part = wp_without_question(sentences_containing_number(word_problem.wp))
    question_part = word_problem.question_part
    return sequence_handler_part(loader, wp_part, 'WP'), sequence_handler_part(loader, question_part, 'QUESTION')


def decide(loader: 'Loader', wp: 'WordProblem') -> Tuple:
    """
    Runs sequence handler and according to some rules it decides what should happen.
    Does not have to work properly for example: Na výlet jelo 20 chlapců a 2 krát méně děvčat. Kolik dětí jelo na výlet?
    It finds only one sequence so it applies only NUM1 / NUM2.
    :param loader: Instance of class Loader.
    :param wp: Instance of class WordProblem.
    :return: Tuple, where first parameter is flag is sequence was found and second is expression that should be applied.
    """
    if len(wp.num_dict_wp) == 2:
        (wp_flag, expr1), (q_flag, expr2) = sequence_handler(loader, wp)
        if wp_flag and not q_flag:
            return True, expr1
        elif not wp_flag and q_flag:
            return True, expr2
        elif wp_flag and q_flag:
            return True, expr2.replace('NUM2', expr1)
        else:
            return False, ''
    else:
        return False, ''
