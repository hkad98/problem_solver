from typing import TYPE_CHECKING, Dict, Set

if TYPE_CHECKING:
    from loader import Loader
    from word_problem import WordProblem


def merge_dict(dst: Dict, src: Dict, weight: int = None):
    """
    Merges 2 dictionaries according to some rules.
    :param dst: Destination dictionary.
    :param src: Source dictionary.
    :param weight: Integer weight.
    """
    for value in src:
        if value in dst:
            dst[value] += src[value] if weight is None else weight * src[value]
        else:
            dst[value] = src[value] if weight is None else weight * src[value]


def merge_points(dst: Dict, src: Dict, weight: int = None):
    """
    Mergers 2 dictionaries according to some rules.
    :param dst: Destination dictionary.
    :param src: Source dictionary.
    :param weight: Integer weight.
    """
    for value in src:
        if value in dst:
            dst[value] += src[value] if weight is None else weight * src[value]


def process_wc(dict_important: Dict, word_set: Set, tag: str) -> Dict:
    """
    :param dict_important: Dictionary of "important" words for specific word class.
    :param word_set: Set of words of specific word class.
    :param tag: String. WP or QUESTION
    :return:
    """
    result = {}
    for word in word_set:
        if word in dict_important and tag in dict_important[word]:
            merge_dict(result, dict_important[word][tag])
    return result


def process_part(word_problem: 'WordProblem', tag: str, wc_important: Dict) -> Dict:
    """
    Process concrete part of word problem. QUESTION or WP.
    :param word_problem: Instance of class WordProblem.
    :param tag: String representing part of word problem which will be processed.
    :param wc_important: Dictionary of important word classes.
    :return: Returns dictionary where KEY is tag ('WP','QUESTION') and VALUE is dictionary where KEY is an expression and VALUE is number.
    Return example:
    'WP':
        'NUM1 + NUM1 / NUM2':
            -34
        'NUM1 + NUM1 * NUM2':
            -14
    """
    assert tag == 'WP' or tag == 'QUESTION'
    res = {tag: {}}
    wp_part = word_problem.wc_question_free if tag == 'WP' else word_problem.wc_question
    for word_class in wp_part.keys():
        if word_class in wc_important:
            ret = process_wc(wc_important[word_class], wp_part[word_class], tag)
            merge_dict(res[tag], ret, word_problem.weights[tag][word_class])
    return res


def sanity_check(expr_dictionary: Dict, possible: Set):
    """
    For all expression where would be result non positive integer sets 0.
    :param expr_dictionary: Dictionary of expressions with all value numbers.
    :param possible: Expressions that can be classified - they are in train set.
    """
    for e in expr_dictionary:
        if e not in possible:
            expr_dictionary[e] = 0


def workflow(word_problem: 'WordProblem', loader: 'Loader') -> Dict:
    """
    Rums pre-defined workflow steps.
    :param word_problem: Instance of class WordProblem.
    :param loader: Instance of class Loader.
    :return: Returns dictionary where key is expression and value is number, representing weight of expression.
    """
    init = word_problem.init_dictionary.copy()
    wp = process_part(word_problem, 'WP', loader.important_wc)
    q = process_part(word_problem, 'QUESTION', loader.important_wc)
    merge_points(init, wp['WP'])
    merge_points(init, q['QUESTION'])
    sanity_check(init, word_problem.possible_expressions)
    return init
