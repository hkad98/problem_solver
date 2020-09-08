import os
import load
import sentence_functions as s
import help_functions as hf
import re
import numpy as np

from typing import List, Tuple

dir_temporary = "temporary"
enc = 'utf8'


def is_model_presented() -> Tuple:
    """
    Checks if model is presented. If model is presented it returns the newest (last by alphabet).
    :return: Tuple. If model is presented it returns filename and True, else '' and False.
    """
    files = np.array(os.listdir(), dtype=object)
    extension_mask = [file.lower().endswith('.udpipe') for file in files]
    flag = any(extension_mask)
    model_filename = files[extension_mask].tolist()[-1] if flag else ''
    return model_filename, any(extension_mask)


def delete_temp_files(directory: str):
    """
    Deletes temporary files.
    :param directory: String. Directory where will be temporary files deleted.
    """
    os.remove('./' + directory + '/results.txt')
    os.remove('./' + directory + '/word_problems.txt')
    os.remove('./' + directory + '/word_problems_conllu.conllu')


def split_values(filename: str) -> Tuple:
    """
    Opens file where are word problems saved and split every line by '|' - result is 3 values: word problem, result, expression.
    :param filename: String. Path where word problems are saved.
    :return: List of [word_problems, results, expressions]
    """
    with open(filename, "r", encoding=enc) as f:
        data = f.read()
    wp, results, expressions = '', '', ''
    lines = data.split('\n')
    for line in lines:
        splited = line.split(' | ')
        wp += splited[0] + '\n'
        results += splited[1] + '\n'
        expressions += splited[2] + '\n'
    wp = wp[:-1]
    results = results[:-1]
    expressions = expressions[:-1]
    return wp.split('\n'), results.split('\n'), expressions.split('\n')


def save_value(directory: str, array: List, value: str, start: int = 1):
    """
    Save values in array to directory, labeling with start and contain value in name.
    :param directory: Directory specifying where will be result saved.
    :param array: List[string]. List containing values.
    :param value: String. Specifying value used in name.
    :param start: Integer. Optional parameter specifying start of counting.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, result in enumerate(array, start=start):
        f = open('./' + directory + '/word_problem' + str(i) + value + '.txt', 'w+')
        f.write(result)
        f.close()


def save_results(directory: str, results: List, start: int = 1):
    """
    Saves results to files.
    :param start: Integer. Optional parameter specifying number of word problem.
    :param directory: Directory specifying where will be result saved.
    :param results: List[string]. List of results.
    """
    save_value(directory, results, 'result', start)


def save_expressions(directory: str, expressions: List, start: int = 1):
    """
    Saves expressions to files.
    :param start: Integer. Optional parameter specifying number of word problem.
    :param directory: Directory specifying where will be expression saved.
    :param expressions: List[string]. List of expressions.
    """
    save_value(directory, expressions, 'expression', start)


def save_word_problems(directory: str, word_problems: List, start: int = 1):
    """
    It saves word problems from list.
    :param start: Integer. Optional parameter specifying number of word problem.
    :param directory: String. Directory specifying where will be word problem saved.
    :param word_problems: List[string]. Word problems in list.
    """
    for i, wp in enumerate(word_problems, start=start):
        with open('./' + directory + '/word_problem' + str(i) + '.txt', 'w+', encoding=enc) as f:
            f.write(wp)


def save_single(directory: str, number: int, wp: str):
    """
    Saves word problem in CONLLU.
    :param directory: String. Directory specifying where will be word problem saved.
    :param number: Integer. Number representing the number of word problem in directory.
    :param wp: String. Word problem in CONLLU.
    """
    f = open('./' + directory + '/word_problem' + str(number) + 'conllu.conllu', 'w+')
    f.write(wp)
    f.close()


def parse_single_conllu(word_problem: str):
    """
    Parse word problem in CONLLU and makes proper labeling.
    :param word_problem: String. Word problem in CONLLU.
    :return: String. Parsed and proper labeled word problem in CONLLU.
    """
    start = "# newdoc\n# newpar\n"
    splited = re.split('# sent_id = .*\n', word_problem)
    splited = [i for i in splited if i != '']
    for i, sentence in enumerate(splited, start=1):
        start += '# sent_id = ' + str(i) + '\n' + sentence
    return start


def parse_conllu(directory: str, path: str, start: int = 1):
    """
    Parsing and saving of CONLLU file containing all word problems.
    :param start: Integer. Optional parameter specifies start of counting word problems. Used when update.
    :param directory: String. Directory where will be CONLLU file save.
    :param path: String. Path to file containing all word problems in CONLLU.
    """
    f = open(path, "r")
    data = f.read().split('SpacesAfter=\\n')
    f.close()
    data = data[:-1]
    first = True
    for counter, wp in enumerate(data, start=start):
        wp += 'SpacesAfter=\\n' + '\n\n'
        if first:
            first = False
            save_single(directory, counter, wp)
        else:
            wp = wp[2:]
            save_single(directory, counter, parse_single_conllu(wp))


def preprocessing(directory: str, wp: str, results: str, model_filename: str, start: int = 1):
    """
    Handles reduction and split of word problems.
    :param model_filename: Specifying name of udpipe model.
    :param directory: String. Directory where will be CONLLU file save.
    :param wp: String. Word problems separated by '\n'.
    :param results: String. Results separated by '\n'.
    :param start: Integer. Optional parameter specifies start of counting word problems. Used when update.
    """
    wps_conllu = process_conllu(directory, [wp, results], model_filename)
    reduced = reduce_word_problem(wps_conllu)
    wps_conllu = process_conllu(directory, [reduced, results], model_filename)

    reduced = reduce_word_problem(wps_conllu)
    wps_conllu = process_conllu(directory, [reduced, results], model_filename)

    parse_conllu(directory, wps_conllu, start)
    delete_temp_files(directory)


def create_full_structure(filename: str, directory: str):
    """
    Workflow of steps creating full structure.
    First it creates lite structure and then it creates CONLLU.
    Use example:
                create_full_structure('./dataset/WP150/testdata.txt', './dataset/WP150/testdata')
    ***
    UDPipe model is  required for full structure - 'czech-pdt-ud-2.5-191206.udpipe' or newer version.
    This model can be download from: https://universaldependencies.org/
    If file is not presented the lite structure is created.
    :param filename: String. Path. Specifying path to file with word problems.
    :param directory: String. Directory where will be CONLLU file save.
    """
    hf.clear_directory(directory)
    model_filename, flag = is_model_presented()
    wp, results = create_lite_structure(filename, directory)
    if flag:
        preprocessing(directory, wp, results, model_filename)
    else:
        print(
            'WARNING: Full structure was required but udpipe model is not presented. Lite structure was created instead.')


def create_lite_structure(filename: str, directory: str) -> Tuple:
    """
    Create structure of word problems for computing. For every word problem it creates 3 files.
    First file is contains word problem.
    Second file contains result of word problem.
    Third file contains expression of word problem.
    :param filename: String. The name of file in which are word problems.
    :param directory: String. Directory, where will be structure saved.
    """
    wp, results, expressions = split_values(filename)
    save_results(directory, results)
    save_expressions(directory, expressions)
    save_word_problems(directory, wp)
    return '\n'.join(wp), '\n'.join(results)


def process_conllu(directory: str, word_problems: List, model_filename: str) -> str:
    """
    It creates file in CONLLU format.
    :param model_filename: Specifying name of udpipe model.
    :param directory: String. Directory where will be CONLLU file save.
    :param word_problems: String. Word problems.
    :return: String. Directory and path to file.
    """
    wps = "./" + directory + "/word_problems.txt"
    wps_conllu = "./" + directory + '/word_problems_conllu.conllu'
    results = "./" + directory + "/results.txt"
    f = open(wps, "w+")
    f.write(word_problems[0])
    f.close()
    f = open(results, "w+")
    f.write(word_problems[1])
    f.close()
    cmd = './udpipe --tokenize --tag --parse ' + model_filename + ' < ' + wps + ' > ' + wps_conllu
    os.system(cmd)
    return wps_conllu


def reduce_word_problem(path: str) -> str:
    """
    Opens file containing word problems in CONLLU format. Data are parsed and reduced.
    :param path: String. Path to file.
    :return: String. Word problems separated by '\n'.
    """
    f = open(path, "r")
    data = f.read().split('SpacesAfter=\\n')
    f.close()
    result = ""
    for wp in data[:-1]:
        wp += 'SpacesAfter=\\n'
        sentences = [[[k for k in j.split('\t')] for j in i.split('\n')] for i in wp.split('\n\n') if
                     len(i) > 1]
        sentences = [sentence for sentence in sentences if len(sentence) > 1]
        result += reduce(sentences)
    return result[:-1]


def reduce(wp: List) -> str:
    """
    It reduces from word problem sequences and if it is possible it splits sentences by character 'a'.
    :param wp: String. Word problem in CONLLU format.
    :return: String. Word problem split or substituted.
    """
    result = ""
    for sentence in wp:
        removed = s.remove_sequences(sentence, wp, load.load_sequences_sa())
        split = s.split_sentence(removed) if s.contains_number(removed) else []
        if len(split) == 0:
            if len(removed) != 0:
                result += s.construct_sentence(removed) + " "
            else:
                result += s.construct_sentence(sentence) + " "
        elif len(split) != 0 and len(split[0]) != 0:
            if len(split[0]) == 0:
                print(split)
            result += s.construct_sentence(split[0]) + ". " if split[0][-1][1] != '.' else s.construct_sentence(
                split[0]) + " "
            result += s.construct_sentence(split[1]) + ". " if split[1][-1][1] != '.' else s.construct_sentence(
                split[1]) + " "
        else:
            result += s.construct_sentence(sentence) + " "
    return result[:-1] + '\n'


def update(filename: str, directory: str):
    """
    Checks if there are new word problems in file specified by filename (compares count with directory).
    If there are new word problems, it updates directory with new word problems. If not nothing happens.
    :param filename: Path. Specifying path to file with word problems.
    :param directory: Path. Specifying path to file where is full structure saved.
    """
    model_filename, flag = is_model_presented()
    if flag:
        wp, results, expressions = split_values(filename)
        count = load.count_in_dir(directory)
        file_size = len(wp)
        if file_size > count:
            wp = wp[count:]
            results = results[count:]
            expressions = expressions[count:]
            save_results(directory, results, start=count + 1)
            save_expressions(directory, expressions, start=count + 1)
            save_word_problems(directory, wp, start=count + 1)
            preprocessing(directory, '\n'.join(wp), '\n'.join(results), model_filename, start=count + 1)
    else:
        print(
            'WARNING: Full structure was required but udpipe model is not presented. Lite structure was created instead.')

# INFO: Use example.
# if __name__ == '__main__':
#     create_full_structure('./dataset/WP500/testdata.txt', './dataset/WP500/testdata')
