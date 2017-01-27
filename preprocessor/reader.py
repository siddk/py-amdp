"""
reader.py
"""
from nltk.tokenize import word_tokenize
import sys
from itertools import izip

numbers = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
           8: "eight", 9: "nine", 10: "ten", 90: "ninety", 180:"one hundred eighty"}


def get_tokens(file_name):
    """
    Returns list of sentences, where each sentence is represented as a list of tokens.
    """
    out_list = []
    with open(file_name, 'r') as f:
        for line in f:
            out_list.append(line.split())
    return out_list

def has_any(words, checks):
    for c in checks:
        if c in words:
            return True
    return False

def clean_L0(lines):
    return filter(lambda x: not has_any(x[0].split(), ['room', 'door', 'robot']), lines)

def clean_L1(pairs):
    ret = []
    for line, ml_line in pairs:
        split_line = line.split()
        if has_any(split_line, ['north', 'south', 'east', 'west', 'step', 'up', 'down', 'right', 'left','robot']):
            continue
        if 'room' in split_line and 'door' not in split_line:
            continue
        if 'door' in split_line and 'room' not in split_line:
            continue
        if 'room' not in split_line and 'door' not in split_line:
            continue
        ret.append((line, ml_line))
    return ret

def clean_L2(lines):
    return filter(lambda x: not has_any(x[0].split(), ['north','south','east','west','step', 'up', 'down', 'right', 'left', 'door', 'robot']), lines)


def clean(lines, in_file):
    if 'L0' in in_file:
        return clean_L0(lines)
    elif 'L1' in in_file:
        return clean_L1(lines)
    elif 'L2' in in_file:
        return clean_L2(lines)
    else:
        raise Exception('Unknown level specified')


def parse(in_file, out_file):
    """
    Parses an input file, writes parsed output to out_file.
    """
    out = []
    in_ml_file = in_file[:-2] + 'ml'
    with open(in_file, 'r') as f, open(in_ml_file, 'r') as f_ml:
        for line, ml_line in izip(f, f_ml):
            toks = word_tokenize(line)
            toks = [x.lower() if not x.isdigit() else numbers[int(x)] for x in toks]
            if toks[-1] != '.':
                toks.append('.')
            out.append((" ".join(toks[:len(toks)-1]), ml_line))

    #out = clean(out, in_file)

    out_ml_file = out_file[:-2] + 'ml'
    with open(out_file, 'w') as f, open(out_ml_file, 'w') as f_ml:
        for line, ml_line in out:
            f.write(line + "\n")
            f_ml.write(ml_line)


if __name__ == "__main__":
    args = sys.argv
    i, o = args[1], args[2]
    parse(i, o)
