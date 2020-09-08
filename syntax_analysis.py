import sentence_functions as s
import help_functions as h
import dictonary_functions as d

from typing import List, Dict, TYPE_CHECKING, Tuple, Set

if TYPE_CHECKING:
    from loader import Loader


def dict_key_look_around(sentence: List, key: str) -> str:
    """
    Looks around KEY - should be NOUN or PROPN and looks around for ADJ.
    :param sentence: List of sentence in CONLLU data.
    :param key: String word - should be NOUN or PROPN.
    :return: Returns input KEY if nothing was found else KEY + ADJ or ADJ + KEY.
    """
    key_index = [word[2] for word in sentence].index(key)
    new_key = key
    if key_index + 1 < len(sentence) and (
            sentence[key_index + 1][3] == 'ADJ' or sentence[key_index + 1][7] == 'orphan' or sentence[key_index + 1][
        7] == 'obj'):
        new_key = new_key + " " + sentence[key_index + 1][2]
    if 0 <= key_index - 1 and (
            sentence[key_index - 1][3] == 'ADJ' or sentence[key_index - 1][7] == 'orphan' or sentence[key_index - 1][
        7] == 'obj'):
        new_key = sentence[key_index - 1][2] + " " + new_key
    return new_key


def get_around(sentence: List, idx: str, unwanted_child: int = None, idx_as_child: bool = True) -> str:
    """
    Gets children by given index - idx.
    :param sentence: List of sentence in CONLLU data.
    :param idx: Given index where children will be look for.
    :param unwanted_child: Index of unwanted child.
    :param idx_as_child: If idx will be placed as a children.
    :return: String result of children.
    """
    children = get_children(sentence, idx, unwanted_child)
    if idx_as_child:
        children += [int(idx)]
    children.sort()
    res = ''
    for i in children:
        res += sentence[i - 1][2] + ' '
    res = res[0:len(res) - 1]
    return res


def get_children(sentence: List, idx: str, unwanted_child: str = None) -> List:
    """
    Gets words where parent is idx.
    :param sentence: List of sentence in CONLLU data.
    :param idx: Given index where children will be look for.
    :param unwanted_child: Index of unwanted child.
    :return: List of index children.
    """
    if unwanted_child:
        res = [int(word[0]) for word in sentence if word[6] == idx and word[0] != unwanted_child]
        for child in res:
            res += get_children(sentence, str(child), unwanted_child)
    else:
        res = [int(word[0]) for word in sentence if word[6] == idx]
        for child in res:
            res += get_children(sentence, str(child))
    return res


def has_children(sentence: List, idx: str, unwanted_child: str = None) -> bool:
    """
    Method checks if word has children.
    :param sentence: List of sentence in CONLLU data.
    :param idx: Index of parent.
    :param unwanted_child: Index of unwanted child.
    :return: Children.
    """
    if unwanted_child:
        # unwanted_child != None
        res = [word for word in sentence if word[6] == idx and word[0] != unwanted_child]
    else:
        # unwanted_child == None
        res = [word for word in sentence if word[6] == idx]
    return len(res) != 0


def create_dictionary(result_dict: Dict, number: str, sentence: List, num_index: str, unwanted: int, subj_index: int,
                      idx_as_child: bool = True,
                      prev_sentence: List = None):
    """
    Adds information about number and its corresponding entity into dictionary.
    :param result_dict: Resulting dictionary, which will be modified.
    :param number: Found number.
    :param sentence: List of sentence in CONLLU data.
    :param num_index: Index of number.
    :param unwanted: Unwanted index.
    :param subj_index: Index of subject in sentence.
    :param idx_as_child: Bool value if idx will be take as children.
    :param prev_sentence: List of previous sentence in CONLLU data.
    """
    sub_dict = dict()
    sub_dict[get_around(sentence, num_index, unwanted, idx_as_child=idx_as_child)] = number
    if prev_sentence is None:
        result_dict[get_around(sentence, str(subj_index + 1))] = sub_dict
    else:
        result_dict[get_around(prev_sentence, str(subj_index + 1))] = sub_dict


def process_verb_sign(sentence: str, number: str, wc_lemma_dict: Dict) -> str:
    """
    Finds nearest verb to number and by dictionary it decides what operation should be applied to number.
    :param sentence: List of sentence in CONLLU data.
    :param number: String representation of number.
    :param wc_lemma_dict: Dictionary of wc_lemma_dictionary.
    :return: Positive or negative number according to dictionary.
    """
    file = open('./database/verb_sign.txt', 'r').read()
    dictionary = eval(file)
    verb_dict = s.find_nearest_wc(wc_lemma_dict, sentence, 'VERB')
    if len(verb_dict) != 0:
        if verb_dict['NUM1'] in dictionary['+']:
            return number
        elif verb_dict['NUM1'] in dictionary['-']:
            return '-' + number
        else:
            return number
    else:
        return number


def dict_by_tree(sentence: List, data: List, i: int, loader: 'Loader') -> Dict:
    """
    Extract important information from sentence according to some rules.
    :param sentence: Extracted sentence.
    :param data: CONLLU data.
    :param i: Index of sentence. It is needed for looking for entities in previous sentences.
    :param loader: Instance of class Loader.
    :return: Dictionary of important information.
    """
    nums = [num for num in s.find_word_class(sentence, {'NUM'}) if 'NumForm=Word' not in num[5].split('|')]
    nouns = s.find_word_class(sentence, {'NOUN', 'PROPN'})
    res = {}
    sentence_string = s.construct_tokenized([v[1] for v in sentence])
    if len(nums) == 0:
        return res
    elif len(nums) == 1:
        num_index = nums[0][0]
        number = nums[0][1]
        parent_index = nums[0][6]
        parent = sentence[int(parent_index) - 1]
        if parent_index == '0':
            # INFO: wrong CONLLU format or wrong substitution NUMBER cannot be root
            return res
        else:
            number = process_verb_sign(sentence_string, number, loader.wc_lemma_dictionary)
            if parent[7] == 'nsubj':
                # INFO: parent is subject, KEY is parent + child
                res[get_around(sentence, parent_index, num_index)] = number
            elif parent[7] != 'nsubj' and 'nsubj' in [noun[7] for noun in nouns]:
                # INFO: parent is not subject, but subject is in the sentence
                index = [word[7] for word in sentence].index('nsubj')
                if parent[3] != 'NOUN':
                    create_dictionary(res, number, sentence, num_index, unwanted=index, subj_index=index,
                                      idx_as_child=False)
                else:
                    create_dictionary(res, number, sentence, parent_index, unwanted=num_index, subj_index=index)
            else:
                # INFO: parent is not subject and subject is not in sentence
                # INFO: subject is taken from the previous sentence if not found ignore it
                prev_sentence = [word for word in data[i - 1] if len(word) > 1] if i != 0 else []
                if 'nsubj' in [word[7] for word in prev_sentence]:
                    prev_nsubj = [word[7] for word in prev_sentence].index('nsubj')
                    if parent[3] != 'NOUN':
                        create_dictionary(res, number, sentence, num_index, unwanted=prev_nsubj, subj_index=prev_nsubj,
                                          idx_as_child=False, prev_sentence=prev_sentence)
                    else:
                        create_dictionary(res, number, sentence, parent_index, unwanted=num_index,
                                          subj_index=prev_nsubj,
                                          prev_sentence=prev_sentence)
                else:
                    res[get_around(sentence, parent_index, num_index)] = number
    else:
        # INFO: subordinate clause or more numbers
        if s.contains_symbol(sentence, ',') and len(s.filter_numbers_sentence(sentence)) == 2:
            nums = [int(num[1]) for num in nums]
            res[' '] = max(nums) - min(nums)
            return res
        # INFO: cannot handle more than 3 numbers in one sentence there was no word problem where this would occurred so there was no analysis
        # INFO: if there is 2 numbers and sentence does not contain ',', there was wrong substitution
        return res
    return res


def merge_dict(src: Dict, dst: Dict):
    """
    Merging two dictionaries into dst.
    :param src: Source dictionary.
    :param dst: Destination dictionary.
    """
    for key in src.keys():
        if key not in dst:
            dst[key] = src[key]
        else:
            value1 = src[key]
            value2 = dst[key]
            if isinstance(value1, dict) and isinstance(value2, dict):
                merge_dict(src[key], dst[key])
            elif isinstance(value1, str) and isinstance(value2, str):
                try:
                    dst[key] = (int(value1) + int(value2))
                except ValueError:
                    dst[key] = 1


def get_dictionary(data: List, loader: 'Loader') -> Dict:
    """
    Handles creating dictionary.
    :param data: List of list containing parsed CONLLU string by \n.
    :param loader: Instance of class Loader.
    :return: Dictionary of important information.
    """
    result = {}
    sentence_idx = 0
    for i in data:
        sentence = [x for x in i if len(x) > 3]
        dic = dict_by_tree(sentence, data, sentence_idx, loader)
        merge_dict(dic, result)
        sentence_idx += 1
    return result


def check_shortcuts(sentences: List, count: int, data: str) -> Tuple:
    """
    Checks if shortcut exists.
    :param sentences: List of all sentences in word problem (including question part).
    :param count: Integer specifying the number of sentences.
    :param data: Word problem represented in raw CONLLU foramt.
    :return: Tuple. If shortcut exists it returns True, result. If shortcut does not exists it returns True, result.
    """
    numbers = s.filter_numbers_sentences(sentences)
    if len(numbers) == 2:
        question = s.get_sentence(count, data)
        sentences = s.get_sentences(data, count + 1)
        minus, plus = h.database_word_handler(question)
        if minus:
            return True, d.shortcut_minus(numbers)
        if plus:
            return True, d.shortcut_plus(numbers)
        if h.contains_other(sentences):
            return True, d.shortcut_minus(numbers)
    else:
        return False, -1
    return False, -1


def sum_up_leaves(entity_dict: Dict) -> int:
    """
    Moves down in entity dictionary when it reaches floor where all keys are numbers it sums them up.
    :param entity_dict: Dictionary of important words and relations.
    :return: If it is possible it returns sum up of floor numbers else -1.
    """
    if len(entity_dict) == 0:
        return -1
    if isinstance(entity_dict, dict):
        values = list(entity_dict.values())
        if any(map(lambda x: not isinstance(x, dict), values)):
            result = 0
            for v in values:
                if not isinstance(v, dict):
                    result += int(v)
            return result
        else:
            if len(values) == 1:
                return sum_up_leaves(values[0])
            else:
                new_dict = dict()
                for v in values:
                    merge_dict(v, new_dict)
                return sum_up_leaves(new_dict)
    else:
        return -1


def process_entity_dict_hard(words: Set, entity_dict: Dict) -> int:
    """
    Finds words in entity dictionary. If nouns are found it gets their value checks if it is dictionary or a string and then process it.
    :param words: Set of words.
    :param entity_dict: Created dictionary structure of important words, relations and numbers in leaves.
    :return: If results was found it returns result else -1.
    """
    new_entity_dict = dict()
    used_nouns = set()
    for word in words:
        if word in entity_dict:
            used_nouns.add(word)
            if isinstance(entity_dict[word], dict):
                merge_dict(entity_dict[word], new_entity_dict)
            else:
                return int(entity_dict[word])
    return process_entity_dict_hard(words - used_nouns, new_entity_dict) if len(new_entity_dict) != 0 else -1


def process_entity_dict_easy(words: Set, entity_dict) -> int:
    """
    More benevolent than process_entity_dict_hard. Allows word by substring of KEY.
    :param words: Set of words.
    :param entity_dict: Created dictionary structure of important words, relations and numbers in leaves.
    :return: If results was found it returns result else -1.
    """
    result = 0
    if isinstance(entity_dict, dict):
        for word in words:
            for key in entity_dict.keys():
                if word in key and isinstance(entity_dict[key], str):
                    try:
                        result += int(entity_dict[key])
                    except ValueError as e:
                        continue
        return result if result != 0 else -1
    else:
        try:
            # WHEN MESSY DATA GETS IN THIS SHOULD HANDLE THEM
            result = int(entity_dict[0]) if isinstance(entity_dict, list) else int(entity_dict)
        except ValueError as e:
            return -1
        return result


def process_entity_dict(words: List, entity_dict: Dict) -> int:
    """
    Runs both process.
    :param words: Set of words.
    :param entity_dict: Created dictionary structure of important words, relations and numbers in leaves.
    :return: If results was found it returns result - prefer result_hard else -1.
    """
    result_hard = process_entity_dict_hard(set(words), entity_dict)
    result_easy = process_entity_dict_easy(set(words), entity_dict)
    if result_hard == -1 and result_easy == -1:
        return -1
    else:
        return result_hard if result_hard != -1 else result_easy


def solve(data: str, loader: 'Loader') -> Tuple:
    """
    Solve word problem given in parameter data by syntax analysis.
    :param data: String representing CONLLU data loaded from file.
    :param loader: Instance o class Loader.
    :return: If result is found it returns result else -1.
    """
    count = s.sentence_count(data)
    nouns = [x[2] for x in s.find_word_class(s.get_sentence(count, data), {'NOUN'})]
    propn = [x[2] for x in s.find_word_class(s.get_sentence(count, data), {'PROPN'})]
    sentences = s.get_sentences(data, count)
    found_flag, shortcut_result = check_shortcuts(sentences, count, data)
    if found_flag:
        return shortcut_result, True
    entity_dict = get_dictionary(sentences, loader)
    sentence = [word for word in s.get_sentence(count, data) if len(word) > 3]
    nouns = [dict_key_look_around(sentence, noun) for noun in nouns]
    nouns += propn
    values = list(entity_dict.values())
    entity_dict = values[0] if len(entity_dict.keys()) == 1 and isinstance(values[0], dict) else entity_dict
    result = process_entity_dict(nouns, entity_dict)
    return (result, False) if result != -1 else (sum_up_leaves(entity_dict), False)
