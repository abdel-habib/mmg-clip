import argparse
import numpy as np
from prettytable import PrettyTable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', type=str, help='Path to the text file.`.', required=True)

    args = parser.parse_args()

    if not args.file_path.endswith('.txt'):
        raise ValueError("File path should be a path of a text `.txt` file.") 

    with open(args.file_path) as file:
        len_list = []
        n_sentences_list = []

        results = PrettyTable(["Description", "Value", "Index"])
        results_general = PrettyTable(['General'])

        for line in file:
            line = line.rstrip()[1:-1] # remove "" that wraps the text
            len_list.append(len(line.split()))
            n_sentences_list.append(len(line.split('.')) - 1)

        results_general.add_row([f'Total number of reports is {len(len_list)}'])
        results_general.add_row([f'Total number of unique reports is {len(np.unique(len_list))}'])
        results_general.add_row([f'Average count of words in all reports is {round(np.mean(len_list), 3)}, STD is {round(np.std(len_list), 3)}'])

        results.add_row(["Minimum count of words in one report", min(len_list), len_list.index(min(len_list))+1])
        results.add_row(["Maximum count of words in one report", max(len_list), len_list.index(max(len_list))+1])
        results.add_row(["Minimum count of sentences in one report", min(n_sentences_list), n_sentences_list.index(min(n_sentences_list))+1])
        results.add_row(["Maximum count of sentences in one report", max(n_sentences_list), n_sentences_list.index(max(n_sentences_list))+1])

        print(results_general)
        print(results)