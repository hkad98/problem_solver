import numpy as np
import load
import sentence_functions as sf

from typing import Dict, Set

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


def get_template(word_problem: str, wc_lemma_dict: Dict, wanted_wc: Set = None, tokenized_flag: bool = False,
                 start=1):
    """
    Converts word problem into template. Changes words it their word class is in wanted wc.
    :param start: Optional parameter start. Specifying counting.
    :param tokenized_flag: Boolean representing if return value will be tokenized or not.
    :param word_problem: String word problem.
    :param wc_lemma_dict: Dictionary of pair word class and lemma.
    :param wanted_wc: Set of word classes that are wanted by user.
    :return: String word problem in template format or List if tokenized flag is True.
    """

    def substitute(wp, lemma_identifier, identifier, single=False):
        """
        Substitutes all lemmas in word problem.
        :param wp: Tokenized word problem in lemma format.
        :param lemma_identifier: Lemma identifier specifying what word change.
        :param identifier: What we will change for lemma identifier.
        :param single: Specifies if we change only first occurrence.
        """
        if single:
            wp[np.where(wp == lemma_identifier)[0][0]] = identifier
        else:
            wp[wp == lemma_identifier] = identifier

    added = set()
    word_problem_lemma = np.array(lemmatization(word_problem, wc_lemma_dict, tokenized=True), dtype=object)
    tokenized = sf.tokenize_text(word_problem)
    wanted_wc = {'NOUN', 'ADJ'} if wanted_wc is None else wanted_wc
    wanted_wc.add('NUM')
    counter = {k: start for k in wanted_wc}
    for token in tokenized:
        low = token.lower()
        if low in wc_lemma_dict and low not in added:
            word_class = wc_lemma_dict[low][0]
            lemma = wc_lemma_dict[low][1]
            added.add(low)  # low or lemma ? or both ?
            if word_class in wanted_wc:
                substitute(word_problem_lemma, lemma, word_class + str(counter[word_class]))
                counter[word_class] = counter[word_class] + 1 if isinstance(start, int) else start
        elif token.isnumeric():
            substitute(word_problem_lemma, token, 'NUM' + str(counter['NUM']), single=True)
            counter['NUM'] = counter['NUM'] + 1 if isinstance(start, int) else start
    return sf.construct_tokenized(list(word_problem_lemma)) if not tokenized_flag else word_problem_lemma


def lemmatization(word_problem, wp_lemma_dictionary=None, tokenized=False):
    """
    Given word problem it converts into lemma version.
    :param word_problem: Word problem in String format.
    :param wp_lemma_dictionary: Dictionary of pair word class and lemma.
    :param tokenized: Optional value defining if we want to return tokenized text.
    :return: Tokenized or not tokenized lemma word problem. String or List.
    """
    wp_lemma_dictionary = load.load_wc_lemma_dict() if wp_lemma_dictionary is None else wp_lemma_dictionary
    tokens = sf.tokenize_text(word_problem)
    result = []
    for token in tokens:
        low = token.lower()
        if low in wp_lemma_dictionary:
            result.append(wp_lemma_dictionary[low][1])
        else:
            result.append(token)
    return sf.construct_tokenized(result) if not tokenized else result


# INFO: Use example.
# if __name__ == '__main__':
#     from loader import Loader
#     loader_instance = Loader()
#     print(get_template('Ota má 80 Kč.', loader_instance.wc_lemma_dictionary))
