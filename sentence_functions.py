import copy
import numpy as np
import re

from typing import List, Dict, TYPE_CHECKING, Set

if TYPE_CHECKING:
    from loader import Loader


def has_number(sentence: str) -> bool:
    """
    Checks if sentence - String contains a number.
    https://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number
    :param sentence: Sentence from word problem in String format.
    :return: Boolean: True if sentence contains digit number. False if sentence does not contain digit number.
    """
    return any(c.isdigit() for c in sentence)


def sentences_containing_number(word_problem: str) -> str:
    """
    Returns word problem without redundant sentences.
    :param word_problem: Word problem as String.
    :return: Word problem without redundant sentences.
    """
    question = get_question(word_problem)
    question_free = wp_without_question(word_problem)
    d = '. '
    sentences = [s + d for s in question_free.split(d) if len(s) != 0]
    result = []
    for sentence in sentences:
        if has_number(sentence):
            result.append(sentence)
    return ''.join(result + question)


def create_bucket_dictionary(sentence: str, loader: 'Loader') -> Dict:
    """
    Creates dictionary of word classes and puts tokens from sentence in it.
    :param sentence: Sentence from word problem in String format.
    :param loader: Instance of class Loader from loader file.
    :return: Tokens from sentence are converted to lemmas and placed into buckets where each bucket correspond to one word class.
    """
    dictionary = loader.wc_lemma_dictionary
    tokenized = tokenize_text(sentence)
    result = dict()
    for token in tokenized:
        token = token.lower()
        if token in dictionary:
            word_class = dictionary[token][0]
            lemma = dictionary[token][1]
            if word_class in result:
                result[word_class].add(lemma)
            else:
                result[word_class] = set()
                result[word_class].add(lemma)
    return result


def create_wc_dict(sentence: str, loader: 'Loader') -> Dict:
    """
    Creates dictionary of word classes and puts tokens from sentence in it.
    :param sentence: Sentence from word problem in String format. Sentence is consists of lemmas.
    :param loader: Instance of class Loader from loader file.
    :return: Tokens from sentence are placed into buckets where each bucket correspond to one word class.
    """
    dictionary = loader.wc_lemma_dictionary
    tokenized = tokenize_text(sentence)
    result = dict()
    for token in tokenized:
        token = token.lower()
        if token in dictionary:
            word_class = dictionary[token][0]
            if word_class in result:
                result[word_class].add(token)
            else:
                result[word_class] = set()
                result[word_class].add(token)
    return result


def construct_sentence(sentence: List) -> str:
    """
    Tokenize text from CONLLU sentence and then constructs String with capital char in the beginning.
    :param sentence: List of CONLLU data. Representing word problem sentence.
    :return: Constructed sentence from list in String format.
    """
    tokenized_text = [word[1] for word in sentence if len(word) > 1]
    tokenized_text[0] = tokenized_text[0].capitalize()
    return construct_tokenized(tokenized_text)


def tokenize_text(word_problem: str) -> np.ndarray:
    """
    Tokenize string word by word.
    :param word_problem: Word problem in string format
    :return: Numpy array of tokens
    """
    cpy = word_problem
    cpy = cpy.replace(',', ' ,')
    cpy = re.sub(r'\. ', ' . ', cpy).replace('?', ' ?')
    cpy = np.array(cpy.split(' '))
    return cpy


def get_word_lemma(sentence: List) -> List:
    """
    Gets words lemma from sentence.
    :param sentence: List of CONLLU data representing sentence from word problem.
    :return: List of lemma words.
    """
    return [word[2] for word in sentence if len(word) >= 2]


def filter_numbers_sentence(sentence: List) -> List:
    """
    Filters numbers from sentences (NOT question) for shortcut.
    :param sentence: List of CONLLU data. Representing word problem sentence.
    :return: List of numbers.
    """
    res = []
    nums = find_word_class(sentence, {'NUM'})
    res += [num[1] for num in nums if len(num) > 3 and 'NumForm=Word' not in num[5].split('|')]
    return res


def filter_numbers_sentences(sentences: List) -> List:
    """
    Filters numbers from sentences (NOT question) for shortcut.
    :param sentences:  List of sentences CONLLU.
    :return: List of numbers.
    """
    res = []
    for sentence in sentences:
        nums = find_word_class(sentence, {'NUM'})
        res += [num[1] for num in nums if len(num) > 3 and 'NumForm=Word' not in num[5].split('|')]
    return res


def get_index_word_class(sentence: List, word_class: str, specified: str = None) -> int:
    """
    Gets index by word_class and if is given by specification.
    :param sentence: List of CONLLU data. Representing word problem sentence. Sentence where index will be found.
    :param word_class: Specified word class to be found.
    :param specified: Specified word we are looking for.
    :return: Index of word_class or word_class and specified word we are looking for.
    """
    counter = -1
    found = False
    if specified is None:
        for word in sentence:
            counter += 1
            if len(word) > 3 and word[3] == word_class:
                found = True
                break
    else:
        for word in sentence:
            counter += 1
            if len(word) > 3 and word[3] == word_class and word[2] == specified:
                found = True
                break
    return counter if found else -1


def find_word_class(sentence: List, word_class: Set) -> List:
    """
    This function finds word_class in a sentence.
    :param word_class: Set of specified word class.
    :param sentence: List of CONLLU data. Representing word problem sentence. Sentence where we will find for word_class.
    :return: List of records containing word_classes.
    """
    return [x for x in sentence if len(x) > 3 and x[3] in word_class]


def count_word_class(sentence: List, word_class: str) -> int:
    """
    Finds out the number of word in specified word class.
    :param sentence: List of CONLLU data. Representing word problem sentence.
    :param word_class: Word class.
    :return: Count of words in specified word class.
    """
    return len([x for x in sentence if len(x) > 3 and x[3] == word_class])


def contains_number(sentence: List) -> bool:
    """
    Checks if a sentence contains number specified by NumForm=Digit - 1,2,3,4,...
    :param sentence: List of CONLLU data. Representing word problem sentence.
    :return: Boolean flag if sentence contains number.
    """
    return len(
        [word for word in sentence if len(word) > 1 and word[3] == 'NUM' and 'NumForm=Digit' in word[5].split('|')]) > 0


def contains_symbol(sentence: List, symbol: str) -> bool:
    """
    Checks if sentence contains symbol.
    :param sentence: List of CONLLU data. Representing word problem sentence.
    :param symbol: Symbol can be a word or a character. It needs to be in format 'a', 'ani', 'Pepa', ...
    :return:
    """
    return len([x for x in sentence if len(x) > 2 and x[1] == symbol]) > 0


def contains_a(sentence: List) -> bool:
    """
    Checks if sentence contains conjunction 'a'
    :param sentence: List of CONLLU data. Representing word problem sentence.
    :return: Boolean.
    """
    return contains_symbol(sentence, 'a')


def get_sentences(data: str, count: int) -> List:
    """
    Get number of sentences specified by count.
    :param data: Unformatted data loaded from file.
    :param count: Integer specifying number of sentences.
    :return: Sentences in list.
    """
    return [get_sentence(i, data) for i in range(1, count)]


def get_sentence(x: int, data: str) -> List:
    """
    Get sentence by its sentence_id.
    :param x: Number of wanted sentence.
    :param data: Unformatted String CONLLU data representing whole word problem loaded from file.
    :return: List - sentence.
    """
    found = False
    wanted = "# sent_id = " + str(x)
    res = []
    for line in data.split('\n'):
        if wanted in line and not found:
            found = True
        elif "# sent_id = " in line and found:
            break
        elif found:
            res.append(line.split("\t"))
        else:
            continue
    return res


def sentence_count(data: str) -> int:
    """
    Gets number of sentences. It simply counts spaces between CONLLU sentence representation.
    Possible implementation is to count '# sent id = '
    :param data: Unformatted String CONLLU data representing whole word problem loaded from file.
    :return: Integer. Count of sentences in data.
    """
    lines = [line.split('\t') for line in data.split('\n')]
    result = [line[0].split(" ")[3] for line in lines if len(line[0].split(" ")) == 4]
    return int(result[-1])


def get_numbers_wp(word_problem: str) -> List:
    """
    Get all numbers in word problem.
    :param word_problem: Word problem in string data type
    :return: Numbers in word problem.
    """
    tokens = tokenize_text(word_problem)
    numbers = [token for token in tokens if token.isnumeric()]
    return numbers


def get_numbers_data(data: List) -> List:
    """
    Get all numeric numbers in data.
    :param data: CONLLU data representing all word problem sentences (including question).
    :return: List of all numeric numbers from data.
    """
    res = []
    for i in data:
        res.extend([x[1] for x in i if len(x) > 3 and x[3] == 'NUM' and x[1].isnumeric()])
    return res


def contains_sequence(sentence: List, sequence: List) -> List:
    """
    Checks if a sentence contains specified sequence.
    :param sequence: Specified sequence.
    :param sentence: List of CONLLU data. Representing word problem sentence.
    :return: List of words in sequence. Empty array if not found.
    """
    contains = contains_a(sentence)
    sentence = [word for word in sentence if len(word) > 4]
    res = []
    x = 0
    if contains:
        for word in sentence:
            if word[3] == sequence[x]:
                res.append(word)
                x += 1
                if x == len(sequence):
                    break
            else:
                x = 0
                res = []
    else:
        sequence = sequence[1:]
        for word in sentence:
            if word[3] == sequence[x]:
                res.append(word)
                x += 1
                if x == len(sequence):
                    break
            else:
                x = 0
                res = []
    return res


def change_copy(sentence_copy: List, lines: List, subst, index: str) -> List:
    """
    Helper function for substitute. Removes redundant words. And change the unknown number by known.
    :param sentence_copy: Copy of sentence.
    :param lines: Redundant words.
    :param subst: Substitution.
    :param index: Index of substitute number.
    :return: Changed sentence with new number.
    """
    result = []
    for x in sentence_copy:
        if x[0] not in lines:
            result.append(x)
    for y in result:
        if y[0] == index:
            y[1] = str(int(subst))
            y[2] = str(int(subst))
            break
    return result


def substitute_handler(sentence: List, res: List, data: List) -> List:
    """
    Handler of substitutions. According to size of res it decides which substitution will be applied.
    :param sentence: List of CONLLU data. Representing word problem sentence. Sentence to be changed.
    :param res: Sequence from sentence.
    :param data: CONLLU data representing all word problem sentences (including question).
    :return: Changed sentence with new number.
    """
    if len(res) == 3:
        return substitute_three(sentence, res, data)
    elif len(res) == 4 and res[1][3] == 'NUM' and res[2][3] == 'NUM':
        return substitute_compound(sentence, res, data)
    elif len(res) == 4 and res[2][3] != 'NOUN':
        return substitute_four(sentence, res)
    elif len(res) == 4 and res[2][3] == 'NOUN':
        return substitute_four_optional(sentence, res, data)
    else:
        return substitute_compound(sentence, res, data)


def substitute_three(sentence: List, res: List, data: List) -> List:
    """
    Function for substitution of sequence of size 3.
    :param sentence: List of CONLLU data. Representing word problem sentence.:param sentence: List of CONLLU data. Representing word problem sentence. Sentence to be changed.
    :param res: Sequence from sentence. This will replace part of sentence.
    :param data: CONLLU data representing all word problem sentences (including question).
    :return: Changed sentence with new number.
    """
    sentence_copy = copy.deepcopy(sentence)
    subst = 0
    previous_num = get_numbers_data(data)[0]  # CONSIDER ONLY 2 NUMBERS
    if res[0][3] == 'NUM':
        if res[2][1] == 'méně':
            subst = int(previous_num) / int(res[0][1])
        elif res[2][1] == 'více' or res[2][1] == 'víc':
            subst = int(previous_num) * int(res[0][1])
        lines = [x[0] for x in res if x[3] != 'NUM']
        num = res[0][0]
    else:
        if res[2][1] == 'méně':
            subst = int(previous_num) - int(res[1][1])
        elif res[2][1] == 'více' or res[2][1] == 'víc':
            subst = int(previous_num) + int(res[1][1])
        lines = [x[0] for x in res if x[3] != 'NUM']
        num = res[1][0]
    return change_copy(sentence_copy, lines, subst, num)


def substitute_compound(sentence: List, res: List, data: List) -> List:
    """
    Function for substitution of sequence in other sentence.
    :param sentence: List of CONLLU data. Representing word problem sentence. Sentence to be changed.
    :param res: Sequence from sentence. This will replace part of sentence.
    :param data: CONLLU data representing all word problem sentences (including question).
    :return: Changed sentence with new number.
    """
    sentence_copy = copy.deepcopy(sentence)
    subst = 0
    previous_num = get_numbers_data(data)[0]  # CONSIDER ONLY 2 NUMBERS
    if res[3][1] == 'méně':
        subst = int(previous_num) - int(res[1][1])
    elif res[3][1] == 'více' or res[3][1] == 'víc':
        subst = int(previous_num) + int(res[1][1])
    lines = [x[0] for x in res if x[3] != 'NUM' and x[3] != 'NOUN']
    num = str(int(lines[0]) + 1)
    return change_copy(sentence_copy, lines, subst, num)


def substitute_four(sentence: List, res: List) -> List:
    """
    Function for substitution of sequence of size 4 and other rules.
    :param sentence: List of CONLLU data. Representing word problem sentence. Sentence to be changed.
    :param res: Sequence from sentence. This will replace part of sentence.
    :return: Changed sentence with new number.
    """
    sentence_copy = copy.deepcopy(sentence)
    subst = 0
    if res[1][3] == 'NUM':
        numbers = [x[1] for x in sentence if len(x) > 1 and x[3] == 'NUM']
        if res[3][1] == 'méně':
            subst = int(numbers[numbers.index(res[1][1]) - 1]) / int(res[1][1])
        elif res[3][1] == 'více' or res[3][1] == 'víc':
            subst = int(numbers[numbers.index(res[1][1]) - 1]) * int(res[1][1])
        lines = [x[0] for x in res[1:] if x[3] != 'NUM']
        num = str(int(lines[0]) - 1)
    else:
        numbers = [x[1] for x in sentence if len(x) > 1 and x[3] == 'NUM']
        if res[2][1] == 'méně':
            subst = int(numbers[numbers.index(res[1][1]) - 1]) - int(res[1][1])
        elif res[2][1] == 'více' or res[2][1] == 'víc':
            subst = int(numbers[numbers.index(res[1][1]) - 1]) + int(res[1][1])
        lines = [x[0] for x in res[1:] if x[3] != 'NUM']
        num = str(int(lines[0]) - 1)
    return change_copy(sentence_copy, lines, subst, num)


def substitute_four_optional(sentence: List, res: List, data: List) -> List:
    """
    Function for substitution of sequence of size 4.
    :param data: CONLLU data representing all word problem sentences (including question).
    :param sentence: List of CONLLU data. Representing word problem sentence. Sentence to be changed.
    :param res: Sequence from sentence. This will replace part of sentence.
    :return: Changed sentence with new number.
    """
    sentence_copy = copy.deepcopy(sentence)
    subst = 0
    previous_number = get_numbers_data(data)[0]  # CONSIDER ONLY 2 NUMBERS
    if res[0][3] == 'NUM':
        if res[3][1] == 'méně':
            subst = int(previous_number) / int(res[0][1])
        elif res[3][1] == 'více' or res[3][1] == 'víc':
            subst = int(previous_number) * int(res[0][1])
        lines = [x[0] for x in res if x[3] != 'NUM' and x[3] != 'NOUN']
        num = str(int(lines[0]) - 1)
    else:
        if res[3][1] == 'méně':
            subst = int(previous_number) - int(res[1][1])
        elif res[3][1] == 'více' or res[3][1] == 'víc':
            subst = int(previous_number) + int(res[1][1])
        lines = [x[0] for x in res if x[3] != 'NUM' and x[3] != 'NOUN']
        num = str(int(lines[0]) + 1)
    return change_copy(sentence_copy, lines, subst, num)


def sequences_handler(sequences: List) -> List:
    """
    Process given sequences. If sequence contains optional '(NOUN)' it creates two sequences.
    One containing optional NOUN.
    Second without NOUN.
    :param sequences: List of sequences.
    :return: List of sequences without optional NOUN.
    """
    res = []
    optional_noun = '(NOUN)'
    for sequence in sequences:
        if optional_noun in sequence:
            res.append(
                sequence[0:sequence.index(optional_noun)] + sequence[sequence.index(optional_noun) + 1:len(sequence)])
            res.append(
                sequence[0:sequence.index(optional_noun)] + ['NOUN'] + sequence[
                                                                       sequence.index(optional_noun) + 1:len(sequence)])
        else:
            res.append(sequence)
    return res


def remove_sequences(sentence: List, data: List, sequences: List) -> List:
    """
    Handles removing of sequences in sentence.
    :param sequences: Sequences to be handled.
    :param sentence: Given sentence of CONLLU format to be handled.
    :param data: CONLLU data representing all word problem sentences (including question).
    :return: Modified sentence or same sentence.
    """
    sequences = sequences_handler(sequences)
    res = []
    for sequence in sequences:
        res.extend(contains_sequence(sentence, sequence))
    if len(res) != 0:
        return substitute_handler(sentence, res, data)
    else:
        return sentence


def split_sentence(sentence: List) -> List:
    """
    Splits sentence according to some rules - splits by conjuction 'a' and according to number of verbs in sentence.
    :param sentence: List of CONLLU data. Representing word problem sentence.
    :return: List of two sentences or empty array if split is not possible.
    """
    first_sentence = []
    second_sentence = []
    a_index = get_index_word_class(sentence, 'CCONJ', 'a')
    verb_index = get_index_word_class(sentence, 'VERB')
    verb_count = count_word_class(sentence, 'VERB')
    if verb_index == -1 and a_index != -1:
        verb_index = get_index_word_class(sentence, 'AUX')
        verb_count = count_word_class(sentence, 'AUX')
    if a_index != -1 and verb_index != -1 and verb_count == 1:
        for i in range(0, a_index):
            if len(sentence[i]) > 4:
                first_sentence.extend([sentence[i]])
                if i <= verb_index:
                    second_sentence.extend([sentence[i]])
        for j in range(a_index + 1, len(sentence) - 1):
            second_sentence.extend([sentence[j]])
        return [first_sentence, second_sentence]
    elif a_index != -1 and verb_index != -1:
        first_sentence = sentence[0:a_index]
        second_sentence = sentence[a_index + 1:len(sentence)]
        return [first_sentence, second_sentence]
    else:
        return []


def wp_without_question(word_problem: str) -> str:
    """
    Gets word problem without questions.
    :param word_problem: Word problem as String.
    :return: Word problem without questions as String.
    """
    sentences = [sentence for sentence in word_problem.split('. ') if '?' not in sentence]
    return '. '.join(sentences) + '. '


def get_question(word_problem: str) -> List:
    """
    Get all sentence containing '?'
    :param word_problem: Word problem as String.
    :return: List of questions - sentences, containing '?'.
    """
    question_part = [sentence for sentence in word_problem.split('. ') if '?' in sentence]
    return question_part if question_part.count('?') == 1 else [q + '?' for q in
                                                                question_part[0].replace('? ', '?').split('?')][:-1]


def find_nearest_wc(wc_lemma_dictionary: Dict, sentence: str, word_class: str) -> Dict:
    """
    Finds nearest word to number according to word class.
    :param wc_lemma_dictionary: Dictionary where key is word and value is tuple word class and lemma.
    :param sentence: String. Sentence.
    :param word_class: String. Word class.
    :return: Returns dictionary where key is number identifier (example: NUM1) and value is lemma of word.
    """
    import create_templates as ct
    ret_val = dict()
    template = ct.get_template(sentence, wc_lemma_dictionary, tokenized_flag=True)
    dictionary = wc_lemma_dictionary
    tokenized = tokenize_text(sentence)
    numbers = [i for i, token in enumerate(tokenized) if token.isnumeric()]
    wc = []
    for i, token in enumerate(tokenized):
        if token.lower() in dictionary and dictionary[token.lower()][0] == word_class:
            wc.append(i)
        else:
            wc.append(-np.inf)
    wc = np.array(wc)
    for number in numbers:
        mask = np.ones(len(tokenized)) * int(number)
        result = abs(wc - mask)
        idx = np.argmin(result)
        if tokenized[idx].lower() in dictionary and dictionary[tokenized[idx].lower()][0] == word_class:
            ret_val[template[number]] = dictionary[tokenized[idx].lower()][1]
    return ret_val


def construct_tokenized(tokenized_text: List) -> str:
    """
    Constructs String from tokenized text.
    :param tokenized_text: List of strings.Tokenized text.
    :return: Returns sentence String.
    """
    symbols = [',', '.', '?']
    first = True
    result = ""
    for token in tokenized_text:
        if token in symbols or first:
            result += token
            first = False
        else:
            result += " " + token
    return result
