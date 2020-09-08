from load import load_raw_txt
from loader import Loader
from word_problem import WordProblem
from typing import Tuple
from create_point import workflow
from data import prepare_data
from svm import scale_on_train, decide, train
import ast
from typing import List, Dict

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


def solve_single(word_problem: 'WordProblem') -> int:
    """
    Solves single word problem.
    :param word_problem: Instance of class WordProblem.
    :return: Predicted results according to machine learning classifier.
    """
    word_problem.weights = loaded_weights
    word_problem.init_dictionary = train_data.word_problems[0].init_dictionary
    word_problem.set_possible_expressions(set(train_data.expressions))
    x_test_points = list(workflow(word_problem, loader).values())
    preprocessed_test = scale_on_train(x_train_points, [x_test_points])
    prediction = list(map(lambda x: classifier.predict([x])[0], preprocessed_test))[0]
    inv_expressions = {v: k for k, v in train_data.expression_labels.items()}
    prediction = eval(inv_expressions[prediction], {}, word_problem.num_dict_wp)
    flag, expression = decide(loader, word_problem)
    if flag:
        prediction = eval(expression, {}, word_problem.num_dict_wp)
    return prediction


def process_single(word_problem: str) -> Tuple:
    """
    Process single word problem.
    :param word_problem: String given by user.
    :return: Tuple error_flag - if an error occurred and result.
    """
    error = False
    result = -1
    try:
        wp = WordProblem(word_problem, loader)
        result = solve_single(wp)
    except Exception as e:
        error = True
        print('Něco je špatně se slovní úlohou. Prosím, zkontrolujte slovní úlohu.')
    return error, result


def process_file(path: str):
    """
    Process file by given path. Beware it depends on interpreter you use for loading file.
    :param path: Given absolute path.
    """
    try:
        data = load_raw_txt(path).split('\n')
        for d in data:
            err_flag, result = process_single(d)
            print('Pro slovní úlohu: ')
            print(d)
            if not err_flag:
                print('Programu vyšel výsledek: ' + str(int(result)))
                print('*' * 10)
            else:
                print('Programu nevyšel výsledek.')
                print('*' * 10)
    except FileNotFoundError:
        print('Bohužel, soubor nebyl nalezen. Prosím zkontrolujte cestu k souboru.')


def action_handler(action_type: str) -> bool:
    """
    Handles action given by user.
    :param action_type: Is string action from user.
    :return: Boolean if action_type '3' is given, it returns False and escapes for cycle.
    """
    if action_type == '1':
        print('Zadejte slovní úlohu:')
        word_problem = input()
        err_flag, result = process_single(word_problem)
        if not err_flag:
            print('*' * 10)
            print('Programu vyšel výsledek: ' + str(int(result)))
            print('*' * 10)
        return True
    elif action_type == '2':
        print('Zadejte prosím cestu k souboru:')
        path = input()
        process_file(path)
        return True
    elif action_type == '3':
        print('Konec programu')
        return False
    else:
        print('Špatná volba akce. Zadejte, prosím, validní akci.')
        return True


def get_train_points(word_problems: List['WordProblem'], weights: Dict = None) -> List:
    """
    Converts word problems into points and if weights is not None, it sets weights.
    :param word_problems: List of word problem objects.
    :param weights: Dictionary of weights given for word class.
    :return: List of points.
    """
    if weights is not None:
        for word_problem in word_problems:
            word_problem.weights = weights
    return [list(workflow(wp, loader).values()) for wp in
            word_problems]


def save_classifier():
    """
    Is used for saving classifier as pickle.
    """
    clf = train(x_train_points, train_data.labels)
    with open('./data/classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)


def load_classifier():
    """
    Is used for loading classifier as pickle.
    """
    with open('./data/classifier.pkl', 'rb') as fid:
        clf = pickle.load(fid)
    return clf


if __name__ == '__main__':
    loader = Loader(path='./data/important_wc.p')
    train_data = prepare_data(loader, directory='./data/traindata')
    loaded_weights = ast.literal_eval(load_raw_txt('./data/weights.txt'))
    x_train_points = get_train_points(train_data.word_problems, loaded_weights)
    classifier = load_classifier()
    print(load_raw_txt('./data/info.txt'))
    while True:
        print(load_raw_txt('./data/usage.txt'))
        print('Jakou akci chcete vykonat?')
        action = input()
        if not action_handler(action):
            break
