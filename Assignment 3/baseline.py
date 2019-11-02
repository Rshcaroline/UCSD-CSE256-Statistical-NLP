'''
@Author: 
@Date: 2019-11-01 19:48:16
@LastEditors: Shihan Ran
@LastEditTime: 2019-11-01 20:41:28
@Email: rshcaroline@gmail.com
@Software: VSCode
@License: Copyright(C), UCSD
@Description: 
'''

import sys
from collections import defaultdict
import math
from replace_with_rare_classes import get_rare_word_classes


def read_counts(counts_file, word_tag, word_dict, uni_tag):
    for l in counts_file:
        line = l.strip().split(' ')
        if line[1] == 'WORDTAG':
            word_tag[(line[3], line[2])] = int(line[0])
            word_dict.append(line[3])
        elif line[1] == '1-GRAM':
            uni_tag[(line[2])] = int(line[0])

def word_with_max_tagger(word_tag, word_dict, uni_tag, word_tag_max):
    for word in word_dict:
        max_tag = ''
        max_val = 0.0
        for tag in uni_tag:
            if float(word_tag[(word, tag)]) / float(uni_tag[(tag)]) > max_val:
                max_val = float(word_tag[(word, tag)]) / float(uni_tag[(tag)])
                max_tag = tag
        word_tag_max[(word)] = max_tag

def tag_gene(word_tag_max, out_f, dev_file):
    output_file = open('rare_words_dev_.txt', 'w')
    for l in dev_file:
        line = l.strip()
        if line:
            if line in word_tag_max:
                out_f.write("%s %s\n" % (line, word_tag_max[(line)]))
            else:
                rare_word_class = get_rare_word_classes(line)
                output_file.write('%s\t%s\t%s\n' % (line, rare_word_class, word_tag_max[(rare_word_class)]))
                out_f.write("%s %s\n" % (line, word_tag_max[(rare_word_class)]))
        else:
            out_f.write("\n")
    output_file.close()

def usage():
    print ("""
    python baseline.py [input_train_counts] [input_dev_file] > [output_file]
        Read in counts file and dev file, produce tagging results.
    """)


if __name__ == "__main__":

    if len(sys.argv)!=3: # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    try:
        counts_file = open(sys.argv[1], "r")
        dev_file = open(sys.argv[2], 'r')
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)
        
    word_tag, uni_tag, word_tag_max = defaultdict(int), defaultdict(int), defaultdict(int)
    word_dict = []

    read_counts(counts_file, word_tag, word_dict, uni_tag)
    counts_file.close()
    word_with_max_tagger(word_tag, word_dict, uni_tag, word_tag_max)
    tag_gene(word_tag_max, sys.stdout, dev_file)
    dev_file.close()