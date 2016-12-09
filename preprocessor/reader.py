"""
reader.py
"""
from nltk.tokenize import word_tokenize
import sys

numbers = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
           8: "eight", 9: "nine", 10: "ten"}


def get_tokens(file_name):
    """
    Returns list of sentences, where each sentence is represented as a list of tokens.
    """
    out_list = []
    with open(file_name, 'r') as f:
        for line in f:
            out_list.append(line.split())
    return out_list


def parse(in_file, out_file):
    """
    Parses an input file, writes parsed output to out_file.
    """
    out = []
    with open(in_file, 'r') as f:
        for line in f:
            toks = word_tokenize(line)
            toks = [x.lower() if not x.isdigit() else numbers[int(x)] for x in toks]
            out.append(" ".join(toks))

    with open(out_file, 'w') as f:
        for line in out:
            f.write(line + "\n")


if __name__ == "__main__":
    args = sys.argv
    i, o = args[1], args[2]
    parse(i, o)