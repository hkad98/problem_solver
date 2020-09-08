import load
import numpy as np

from typing import List, Tuple, TYPE_CHECKING, Dict, Set

if TYPE_CHECKING:
    from loader import Loader


def get_expression(num_dict: Dict, result: int, expressions: List) -> str:
    """
    Iterates over all expressions and try them.
    :param num_dict: Dictionary of numbers.
    :param result: The result of word problem.
    :param expressions: List read from file contains all expressions for n - numbers.
    :return: An expression f.e. NUM1 + NUM2
    """
    for expression in expressions:
        if eval(expression, {}, num_dict) == result:
            return expression
    return "Not found."


def remove_symbols(expression: str) -> Set:
    """
    Removes symbols [' / ', ' + ', ' - ', ' * ', ' ) ', ' ( '] from expressions and returns set of numbers. {'NUM1','NUM2'}
    :param expression: Expression as String 'NUM1 / NUM2 + NUM1'.
    :return: Set of identifiers - {'NUM1','NUM2'}.
    """
    symbols = [' / ', ' + ', ' - ', ' * ', ' ) ', ' ( ']
    for symbol in symbols:
        expression = expression.replace(symbol, ' ')
    return set(expression.split(' '))


def get_possible_expressions(expressions: Set, num_dict: Dict) -> Set:
    """
    Checks what expressions can be used for numbers in num_dict. Expression can be used if it contains <= numbers as num_dict.
    And if evaluation of expression on num_dict is >= 0.
    :param expressions: A set of expressions. {'NUM1 / NUM2 + NUM1', 'NUM1 / NUM2',...} to be tested.
    :param num_dict: {'NUM1': 5, 'NUM2': 3}
    :return: Returns set of possible expressions.
    """
    result = set()
    numbers = set(num_dict.keys())
    max_number = max(numbers)
    for e in expressions:
        if max(remove_symbols(e)) <= max_number and eval(e, {}, num_dict) >= 0:
            result.add(e)
    return result


def handle_wp_num(wp_num_count: int, count: int) -> bool:
    """
    Helper for get_chosen_word_problems. Filters word problems by count of numbers in word problem.
    :param wp_num_count: Flag from get_chosen_word_problems. Defines wanted count of numbers in word problem. Default is None - means count does not matter.
    :param count: Count from word problem.
    :return: Boolean. True if condition is satisfied. False if condition is not satisfied.
    """
    if wp_num_count is None:
        return True
    else:
        return wp_num_count == count


def get_chosen_word_problems(loader: 'Loader', directory: str, chosen: List = None, count: int = None) -> Tuple:
    """
    Will return array of chosen word problems. Chosen is meant to be solved by some expression in chosen set().
    :param count: Specifying the number of files for word problems.
    :param directory: Optional parameter directory to word problems.
    :param loader: Instance of class Loader.
    :param chosen: Set of expressions.
    :return: Pair of chosen word problems and results to them.
    """
    word_problems, results, expressions = load.get_files_for_svm(directory, loader, count)
    if chosen is not None:
        chosen_init = expressions == chosen[0]
        for c in chosen[1:]:
            chosen_init = np.logical_or(chosen_init, expressions == c)
        return word_problems[chosen_init], results[chosen_init], expressions[chosen_init]
    else:
        return word_problems, results, expressions
