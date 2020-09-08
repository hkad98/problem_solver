from load import count_in_dir
from loader import Loader
from svm import evaluate_genetic, lda
from heapq import nlargest
import random
import operator
import copy
from itertools import combinations
from data import prepare_data
from typing import List, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from data import Data


def set_weights(word_problems: List, weight: Dict):
    """
    Sets new weight to word problems.
    :param word_problems: List of WordProblem objects.
    :param weight: New weight to be set to WordProblem.
    """
    for wp in word_problems:
        wp.weights = weight


def init_weights(size: int, threshold: int = 10) -> List:
    """
    Initialize new weights.
    :param size: Is the number of weights to be created.
    :param threshold: Default value is 10.
    :return: List of weights.
    """
    result = []
    for i in range(size):
        template = {'QUESTION': {'VERB': 1, 'PREP': 1, 'CONJ': 1, 'ADV': 1, 'PRON': 1},
                    'WP': {'VERB': 1, 'PREP': 1, 'CONJ': 1, 'ADV': 1, 'PRON': 1}}
        for key in template['WP'].keys():
            template['QUESTION'][key] = random.randrange(-threshold, threshold)
            template['WP'][key] = random.randrange(-threshold, threshold)
        result.append(template)
    return result


def crossover(weight: Dict) -> Dict:
    """
    Makes weight crossover.
    :param weight: Weight to be changed by genetic algorithm - crossover.
    :return: New weight.
    """
    new_dictionary = copy.deepcopy(weight)
    random_weight = init_weights(1)[0]
    parts = ['WP', 'QUESTION']
    part_in = random.randrange(0, len(parts))
    part_out = random.randrange(0, len(parts))
    new_dictionary[parts[part_in]] = random_weight[parts[part_out]]
    return new_dictionary


def mutation_single(weight: Dict, threshold: int = 10) -> Dict:
    """
    Function provides single mutation. Randomly choose attribute to be changed and according to threshold in generates random number.
    :param weight: Dictionary represents weights i.e.: {'QUESTION':{'VERB':5, 'ADV':4, ...}}
    :param threshold: Optional parameter (default: 10).
    :return: New weights represented as dictionary.
    """
    new_dictionary = copy.deepcopy(weight)
    word_classes = ['VERB', 'PREP', 'CONJ', 'ADV', 'PRON']
    q_random = random.randrange(0, len(word_classes))
    wp_random = random.randrange(0, len(word_classes))
    new_dictionary['QUESTION'][word_classes[q_random]] += random.randrange(-threshold, threshold)
    new_dictionary['WP'][word_classes[wp_random]] += random.randrange(-threshold, threshold)
    return new_dictionary


def mutation_all(weight: Dict, threshold: int = 10) -> List:
    """
    Function makes mutation on all positions.
    :param weight: Dictionary represents weights i.e.: {'QUESTION':{'VERB':5, 'ADV':4, ...}}
    :param threshold:Optional parameter (default: 10).
    :return:
    """
    result = []
    word_classes = ['VERB', 'PREP', 'CONJ', 'ADV', 'PRON']
    for i in range(5):
        q_dictionary = copy.deepcopy(weight)
        wp_dictionary = copy.deepcopy(weight)
        q_dictionary['QUESTION'][word_classes[i]] += random.randrange(-threshold, threshold)
        wp_dictionary['WP'][word_classes[i]] += random.randrange(-threshold, threshold)
        result.append(q_dictionary)
        result.append(wp_dictionary)
    return result


def update_weights(weights: List, results: Dict, num: int = 5) -> List:
    """
    Creates new weights by crossover, selection and mutation.
    :param weights: List of weights.
    :param results: Dictionary of where the key is weight and value is success rate.
    :param num: Specifies the number of the best result to be mutated.
    :return: Mutated weights.
    """
    highest = nlargest(num, results, key=results.get)
    new_weights = []
    for val in highest:
        dictionary = weights[val]
        new_weights.append(copy.deepcopy(dictionary))  # SELECTION 1
        new_weights.append(crossover(dictionary))  # CROSSOVER 1
        new_weights.extend(mutation_all(dictionary, threshold=100))  # MUTATION OF EVERY WEIGHT 10
    return new_weights


def save_log(population: int, accuracy: str, weight: Dict, solver: str, parameter_c: int):
    """
    Saves progress to file.
    :param parameter_c: The best parameter C.
    :param solver: What type of solver is used when finding weights.
    :param population: The number of population.
    :param accuracy: The best success rate.
    :param weight: The weight.
    """
    with open("./database/logs.txt", "a") as fp:
        population_info = 'Population: ' + str(population) + '\n'
        solver_info = 'Solver: ' + str(solver) + '\n'
        accuracy_info = 'Best accuracy: ' + str(accuracy) + '\n'
        c_info = 'Best C parameter: ' + str(parameter_c) + '\n'
        weights_info = 'Weights are: ' + str(weight) + '\n'
        divider = '_' * 100 + '\n'
        fp.write(population_info + solver_info + accuracy_info + c_info + weights_info + divider)


def wc_permutations(data: 'Data', loader: 'Loader'):
    """
    Tries all permutations of word classes and to every permutation it returns accuracy.
    :param data: Instance of class Data.
    :param loader: Instance of class Loader.
    """
    cpy = copy.deepcopy(loader.important_wc)
    keys = list(cpy.keys())
    for i in range(1, len(keys) + 1):
        comb = combinations(keys, i)
        for c in comb:
            loader.important_wc = {k: cpy[k] for k in c}
            ret_val = evaluate_genetic(data, loader)
            print("combination:", c)
            print("accuracy:", str(ret_val))
            print("-" * 10)


def power_of_ten(a: List) -> List:
    """
    Creates list of power of tens by given list.
    Use example:
    a = [0, 1, 2, 3]
    -> [1, 10, 100, 1000]
    :param a: List of powers
    :return: Powers of ten.
    """
    return [10 ** x for x in a]


def start_genetic(data: 'Data', loader: 'Loader', generation_count: int = 100, solver: str = 'svm'):
    """
    Runs genetic algorithm on data.

    Use example:
    loader = Loader()
    train_dictionary = './dataset/WP500/traindata'
    chosen = ['NUM1 + NUM2', 'NUM1 - NUM2']
    train_data = Data(directory=train_dictionary, loader=loader, chosen=chosen)
    start_genetic(train_data, loader)

    :param data: Instance of class Data, representing train and test data wrapper for genetic algorithm.
    :param solver: Specifies type of solving - svm or lda.
    :param loader: Instance of class Loader.
    :param generation_count: Optional parameter. Default: 100
    """
    weights = init_weights(10)
    max_index, max_accuracy, best_c = 0, 0, 0
    weights[0] = {'QUESTION': {'VERB': 6, 'PREP': -4, 'CONJ': 8, 'ADV': -7, 'PRON': -3},
                  'WP': {'VERB': 6, 'PREP': -10, 'CONJ': 9, 'ADV': -7, 'PRON': 3}}
    parameters_c = power_of_ten([-3, -2, -1, 0])
    for i in range(generation_count):
        weight_accuracy_dict = {}
        weight_c_dict = {}
        for j, w in enumerate(weights, start=0):
            set_weights(data.word_problems, w)
            accuracies = evaluate_genetic(data, loader, parameters_c=parameters_c,
                                          kernel='rbf') if solver == 'svm' else lda(data,
                                                                                    loader)
            max_accuracy = max(accuracies)
            max_accuracy_index = accuracies.index(max_accuracy)
            weight_accuracy_dict[j] = max_accuracy
            weight_c_dict[j] = parameters_c[max_accuracy_index]
        max_index, max_accuracy = max(weight_accuracy_dict.items(), key=operator.itemgetter(1))
        best_c = weight_c_dict[max_index]
        save_log(i, max_accuracy, weights[max_index], solver, best_c)
        weights = update_weights(weights, weight_accuracy_dict, num=10)
        print("Population: " + str(i) + "\nBest accuracy: " + str(max_accuracy) + "\nParameter C: " + str(best_c))
        print('_' * 20)
        i += 1
        if max_accuracy == 1:
            print("FOUND!!!")
            break
    print('... FINISHED')
    print('The best value is: ' + str(max_accuracy))
    print('The best weights are: ' + str(weights[max_index]))


def run_genetic_for_solver():
    """
    Prepares and runs genetic for Solver - user application.
    """
    solver_load = Loader(path='./data/important_wc.p')
    solver_directory = './data/traindata'
    expression_list = None
    solver_train_data = prepare_data(loader=solver_load, chosen=expression_list, directory=solver_directory,
                                     count=count_in_dir(solver_directory, 3))
    start_genetic(solver_train_data, solver_load)

# INFO: Use example.
# if __name__ == '__main__':
#     expression_list = None
#     load = Loader()
#     directory = './dataset/WP500/traindata'
#     train_data = prepare_data(loader=load, chosen=expression_list, directory=directory)
#     start_genetic(train_data, load)
