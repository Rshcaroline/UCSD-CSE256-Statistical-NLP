'''
@Author: Shihan Ran
@Date: 2019-11-02 11:29:23
@LastEditors: Shihan Ran
@LastEditTime: 2019-11-10 16:34:23
@Email: rshcaroline@gmail.com
@Software: VSCode
@License: Copyright(C), UCSD
@Description: 
'''

import sys
from collections import defaultdict
import math
import itertools
import string


def read_counts(counts_file, word_tag, word_dict, ngram_tag):
    for l in counts_file:
        line = l.strip().split(' ')
        if line[1] == 'WORDTAG':
            word_tag[(line[3], line[2])] = int(line[0])
            word_dict.append(line[3])
        else:
            ngram_tag[tuple(line[2:])] = int(line[0])

def efunc(word_tag, word_dict, ngram_tag, x, y):
    """
    e(x|y) = p(x, y) / p(y)
    """
    if x not in word_dict:
        # if all(c in string.punctuation for c in x):
        #     x = '_ALL_PUNCTUATION_'
        # if all(c.isdigit() for c in x):
        #     x = '_ALL_NUMERIC_'
        if any(c.isdigit() for c in x):
            x = '_CONTAIN_NUMERIC_'
        # if x.isupper():
        #     x = '_ALL_CAP_'
        if x[0].isupper():
            x = '_FIRST_CAP_'
        if x[-1].isupper():
            x = '_LAST_CAP_'
        else:
            x = '_RARE_'
    return word_tag[(x, y)] / float(ngram_tag[(y,)])

def qfunc(ngram_tag, v, w, u, z):
    """
    q(z|w, u, v)
    """
    return ngram_tag[w, u, v, z] / float(ngram_tag[w, u, v])

def viterbi(word_tag, word_dict, ngram_tag, word_list):
    word_list = ['*', '*', '*'] + word_list
    tag_set = ('O', 'I-GENE')
    bp_dict = {}
    pi_dict = {(1, '*', '*', '*'): 1}
    
    # solve pi_dict and bp_dict
    for k in range(2, len(word_list)):
        u_set = tag_set
        v_set = tag_set
        w_set = tag_set
        z_set = tag_set
        if k == 2:
            v_set = ('*', )
            u_set = ('*', )
            w_set = ('*', )
        elif k == 3:
            u_set = ('*', )
            w_set = ('*', )
        elif k == 4:
            w_set = ('*', )
            
        # for different (u, v, z), find the optimal w
        for u, v, z in itertools.product(u_set, v_set, z_set):
            e = efunc(word_tag, word_dict, ngram_tag, word_list[k], z)
            candi_list = [((pi_dict[k - 1, w, u, v] * qfunc(ngram_tag, v, w, u, z) * e), w) for w in w_set]
            pi, bp = max(candi_list, key = lambda x: x[0])
            pi_dict[k, u, v, z] = pi
            bp_dict[k, u, v, z] = bp

    # 'STOP' is the last one
    uvz_list = [(pi_dict[len(word_list) - 1, u, v, z] * qfunc(ngram_tag, 'STOP', u, v, z), (u, v, z)) \
        for (u, v, z) in itertools.product(tag_set, tag_set, tag_set)]
    
    tagn_1, tagn_2, tagn = max(uvz_list, key=lambda x:x[0])[1]
    tag_list = [0] * len(word_list)
    tag_list[-3] = tagn_1
    tag_list[-2] = tagn_2
    tag_list[-1] = tagn
    for i in reversed(range(len(tag_list) - 3)):
        tag_list[i] = bp_dict[i + 3, tag_list[i + 1], tag_list[i + 2], tag_list[i + 3]]
    return tag_list[3:]

def tag_gene(word_tag, word_dict, ngram_tag, out_f, dev_file):
    word_list = []
    for l in dev_file:
        line = l.strip()
        if line:
            word_list.append(line)
        else:
            tag_list = viterbi(word_tag, word_dict, ngram_tag, word_list)
            for word, tag in zip(word_list, tag_list):
                out_f.write("%s %s\n" % (word, tag))
            out_f.write('\n')
            word_list = []

def usage():
    print ("""
    python baseline.py [input_train_counts] [input_dev_file] > [output_file]
        Read in counts file and dev file, produce tagging results.
    """)


if __name__ == "__main__":

    # if len(sys.argv)!=4: # Expect exactly one argument: the training data file
    if len(sys.argv)!=3:
        usage()
        sys.exit(2)

    try:
        counts_file = open(sys.argv[1], "r")
        dev_file = open(sys.argv[2], 'r')
        # out_file = open(sys.argv[3], 'w')
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)
        
    word_tag, ngram_tag = defaultdict(int), defaultdict(int)
    word_dict = []

    read_counts(counts_file, word_tag, word_dict, ngram_tag)
    counts_file.close()
    tag_gene(word_tag, word_dict, ngram_tag, sys.stdout, dev_file)
    # tag_gene(word_tag, word_dict, ngram_tag, out_file, dev_file)
    dev_file.close()