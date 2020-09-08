from create_templates import get_template, lemmatization
from sentence_functions import tokenize_text, create_wc_dict, wp_without_question, get_question, \
    sentences_containing_number
from dictonary_functions import create_num_dict_wp
from expression import get_possible_expressions


class WordProblem:
    init_dictionary = None
    possible_expressions = None

    def __init__(self, word_problem, loader=None):
        """
        This class is wrapper for word problem. It does analysis about word problem given by parameter.

        self.wp - stores word problem which is rid off sentences without integer number.
        self.wp_lemma - if loader is given than word problem is transformed into lemma
        self.wp_template - stores template, where some word classes are substituted by identifiers.

        self.init_dictionary - is dictionary, where KEY is an expression and VALUE is set to 0 (IMPORTANT: it is needed for representing word problem as number)

        self.wc_question_free - stores dictionary of part without question, where KEY is ...
        self.wc_question - stores dictionary of question part, where KEY is ...
        self.question_part - stores question part of word problem
        self.wp_part - stores part of word problem without question (IMPORTANT: consider that every word problem has only one question)
        self.wp_template_tokenized - stores tokenized template, where some word classes are substituted by identifiers.
        self.num_dict_wp - creates dictionary, where KEY is an identifier and VALUE is integer

        :param word_problem: Given word problem as String.
        :param loader: Instance of class Loader.
        """
        self.wp = sentences_containing_number(word_problem)
        self.wp_lemma = lemmatization(self.wp, loader.wc_lemma_dictionary) if loader is not None else None
        self.wp_template = get_template(word_problem, loader.wc_lemma_dictionary) if loader is not None else None
        self.wp_template_tokenized = tokenize_text(self.wp_template) if loader is not None else None

        self.question_part = get_question(self.wp)[0]
        self.wp_part = wp_without_question(self.wp)
        self.wc_question_free = create_wc_dict(self.wp_part, loader) if loader is not None else None
        self.wc_question = create_wc_dict(self.question_part, loader) if loader is not None else None
        self.num_dict_wp = create_num_dict_wp(word_problem)

        # INFO: Can be used for MAX heuristic
        # self.weights = {'QUESTION': {'VERB': 1, 'PREP': 1, 'CONJ': 1, 'ADV': 1, 'PRON': 1},
        #                 'WP': {'VERB': 1, 'PREP': 1, 'CONJ': 1, 'ADV': 1, 'PRON': 1}}

        self.weights = {'QUESTION': {'VERB': 18, 'PREP': -54, 'CONJ': 53, 'ADV': -51, 'PRON': -1},
                        'WP': {'VERB': -63, 'PREP': -9, 'CONJ': -8, 'ADV': -7, 'PRON': -6}}

    def update_weights(self, new_weights):
        self.weights = new_weights
        return self

    def set_possible_expressions(self, expressions):
        self.possible_expressions = get_possible_expressions(expressions, self.num_dict_wp)
