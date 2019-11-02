'''
@Author: 
@Date: 2019-11-01 19:58:55
@LastEditors: Shihan Ran
@LastEditTime: 2019-11-01 21:06:03
@Email: rshcaroline@gmail.com
@Software: VSCode
@License: Copyright(C), UCSD
@Description: 
'''

import sys
from collections import defaultdict
import math


threshold = 5

def get_word_counts(corpus_file):
    """
    Read the corpus_file and return word counts
    """
    word_counts = defaultdict(int)

    for l in corpus_file:
        line = l.strip()
        if line:
            linew = line.split(' ')
            if (linew[0]) in word_counts:
                word_counts[(linew[0])] += 1
            else:
                word_counts[(linew[0])] = 1

    return word_counts

def replace_with_rare(corpus_file, output_file, word_counts):
    """
    Read the corpus_file and replace rare words with rare word classes
    """
    op_ = open('rare_words.txt', 'w')
    for l in corpus_file:
        line = l.strip()
        if line:
            linew = line.split(' ')
            if word_counts[linew[0]] < threshold:
                op_.write(linew[0])
                op_.write('\n')
                output_file.write("_RARE_ %s\n" % (linew[1]))
            else:
                output_file.write(line + "\n")
        else:
            output_file.write("\n")
    op_.close()

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

    word_counts = get_word_counts(input)
    input.close()

    try:
        input = open(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)

    replace_with_rare(input, sys.stdout, word_counts)
