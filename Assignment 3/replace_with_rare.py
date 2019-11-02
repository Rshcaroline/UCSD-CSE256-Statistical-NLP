'''
@Author: Shihan Ran
@Date: 2019-11-01 19:58:55
@LastEditors: Shihan Ran
@LastEditTime: 2019-11-01 20:03:57
@Email: rshcaroline@gmail.com
@Software: VSCode
@License: Copyright(C), UCSD
@Description: 
'''
import sys
from collections import defaultdict
import math


def get_rare_words(corpus_file):
    """
    Read the corpus_file and return a list of rare words
    """
    word_counts = defaultdict(int)
    rare_words = []

    for l in corpus_file:
        line = l.strip()
        if line:
            linew = line.split(' ')
            if (linew[0]) in word_counts:
                word_counts[(linew[0])] += 1
            else:
                word_counts[(linew[0])] = 1

    for key in word_counts:
        if word_counts[key] < 5:
            rare_words.append(key)
    return rare_words

def replace_with_rare(corpus_file, output_file, rare_words):
    """
    Read the corpus_file and replace rare words with '_RARE_'
    """
    for l in corpus_file:
        line = l.strip()
        if line:
            linew = line.split(' ')
            if (linew[0]) in rare_words:
                output_file.write("_RARE_ %s\n" % (linew[1]))
            else:
                output_file.write(line + "\n")
        else:
            output_file.write("\n")

def usage():
    print ("""
    python ./replace_with_rare.py [input_file] > [output_file]
        Read in a gene tagged training input file and produce new training file.
    """)

if __name__ == "__main__":

    if len(sys.argv)!=2:  # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    try:
        input = open(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)

    rare_words = get_rare_words(input)
    input.close()

    try:
        input = open(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)

    replace_with_rare(input, sys.stdout, rare_words)
