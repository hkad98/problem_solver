import sentence_functions as s

from typing import Dict, List


def create_num_dict_wp(word_problem: str) -> Dict:
    """
    Creates word problem number dictionary. From given word problem it extracts numbers and creates dictionary, where numbers have identifier.
    Input: 'Tomáš má 2 autíčka a 3 lodě.
    Output: {'NUM1': 2, 'NUM2': 3}
    :param word_problem: Word problem as String.
    :return: Dictionary where key is identifier for number - 'NUM1', value is number.
    """
    numbers = s.get_numbers_wp(word_problem)
    res = {}
    counter = 1
    for number in numbers:
        res['NUM' + str(counter)] = int(number)
        counter += 1
    return res


def shortcut_minus(nums: List) -> int:
    """
    Helper function for shortcut plus.
    :param nums: List of size 2, containing numbers as Strings. ['8','2']
    :return: Integer. Result of math operation +.
    """
    assert len(nums) == 2
    nums = list(map(int, nums))
    return max(nums) - min(nums)


def shortcut_plus(nums: List) -> int:
    """
    Helper function for shortcut minus.
    :param nums: List of size 2, containing numbers as Strings. ['8','2']
    :return: Integer. Result of math operation -.
    """
    assert len(nums) == 2
    nums = list(map(int, nums))
    return max(nums) + min(nums)
