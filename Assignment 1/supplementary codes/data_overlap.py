'''
@Author: 
@Date: 2019-03-26 17:57:16
@LastEditors: Shihan Ran
@LastEditTime: 2019-10-10 21:59:47
@Email: rshcaroline@gmail.com
@Software: VSCode
@License: Copyright(C), UCSD
@Description: The file data.py contains the main function, that reads in all the train, test, and dev files from the archive, trains all the unigram models, and computes the perplexity of all the models on the different corpora.
'''

#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import sys


# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)


def textToTokens(text):
    """Converts input string to a corpus of tokenized sentences.

    Assumes that the sentences are divided by newlines (but will ignore empty sentences).
    You can use this to try out your own datasets, but is not needed for reading the homework data.
    """
    corpus = []
    sents = text.split("\n")
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    count_vect.fit(sents)
    tokenizer = count_vect.build_tokenizer()
    for s in sents:
        toks = tokenizer(s)
        if len(toks) > 0:
            corpus.append(toks)
    return corpus

def file_splitter(filename, seed = 0, train_prop = 0.7, dev_prop = 0.15,
    test_prop = 0.15):
    """Splits the lines of a file into 3 output files."""
    import random
    rnd = random.Random(seed)
    basename = filename[:-4]
    train_file = open(basename + ".train.txt", "w")
    test_file = open(basename + ".test.txt", "w")
    dev_file = open(basename + ".dev.txt", "w")
    with open(filename, 'r') as f:
        for l in f.readlines():
            p = rnd.random()
            if p < train_prop:
                train_file.write(l)
            elif p < train_prop + dev_prop:
                dev_file.write(l)
            else:
                test_file.write(l)
    train_file.close()
    test_file.close()
    dev_file.close()

def read_texts(tarfname, dname):
    """Read the data from the homework data file.

    Given the location of the data archive file and the name of the
    dataset (one of brown, reuters, or gutenberg), this returns a
    data object containing train, test, and dev data. Each is a list
    of sentences, where each sentence is a sequence of tokens.
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz", errors = 'replace')
    for member in tar.getmembers():
        if dname in member.name and ('train.txt') in member.name:
            print('\ttrain: %s'%(member.name))
            train_txt = unicode(tar.extractfile(member).read(), errors='replace')
        elif dname in member.name and ('test.txt') in member.name:
            print('\ttest: %s'%(member.name))
            test_txt = unicode(tar.extractfile(member).read(), errors='replace')
        elif dname in member.name and ('dev.txt') in member.name:
            print('\tdev: %s'%(member.name))
            dev_txt = unicode(tar.extractfile(member).read(), errors='replace')

    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    count_vect.fit(train_txt.split("\n"))
    tokenizer = count_vect.build_tokenizer()
    class Data: pass
    data = Data()
    data.train = []
    for s in train_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.train.append(toks)
    data.test = []
    for s in test_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.test.append(toks)
    data.dev = []
    for s in dev_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.dev.append(toks)
    print(dname," read.", "train:", len(data.train), "dev:", len(data.dev), "test:", len(data.test))
    return data

def learn_unigram(data, verbose=True):
    """Learns a unigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    from lm import Unigram
    unigram = Unigram()
    unigram.fit_corpus(data.train)
    if verbose:
        print("vocab:", len(unigram.vocab()))
        # evaluate on train, test, and dev
        print("train:", unigram.perplexity(data.train))
        print("dev  :", unigram.perplexity(data.dev))
        print("test :", unigram.perplexity(data.test))
        from generator import Sampler
        sampler = Sampler(unigram)
        print("sample 1: ", " ".join(str(x) for x in sampler.sample_sentence([])))
        print("sample 2: ", " ".join(str(x) for x in sampler.sample_sentence([])))
    return unigram

def learn_trigram(data, delta=1/2**(15), smoothing=True, verbose=True):
    """Learns a trigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    from lm import Trigram
    trigram = Trigram(backoff=0.000001, delta=delta, smoothing=smoothing)
    trigram.fit_corpus(data.train)
    if verbose:
        print("vocab:", len(trigram.vocab()))
        # evaluate on train, test, and dev
        print("train:", trigram.perplexity(data.train))
        print("dev  :", trigram.perplexity(data.dev))
        print("test :", trigram.perplexity(data.test))
        from generator import Sampler
        sampler = Sampler(trigram)
        print("sample 1: ", " ".join(str(x) for x in sampler.sample_sentence([])))
        print("sample 2: ", " ".join(str(x) for x in sampler.sample_sentence([])))
    return trigram

def print_table(table, row_names, col_names, latex_file = None):
    """Pretty prints the table given the table, and row and col names.

    If a latex_file is provided (and tabulate is installed), it also writes a
    file containing the LaTeX source of the table (which you can \input into your report)
    """
    try:
        from tabulate import tabulate
        row_format ="{:>15} " * (len(col_names) + 1)
        rows = map(lambda rt: [rt[0]] + rt[1], zip(row_names,table.tolist()))
        rows = list(rows)

        print(tabulate(rows, headers = [""] + col_names))
        if latex_file is not None:
            latex_str = tabulate(rows, headers = [""] + col_names, tablefmt="latex")
            with open(latex_file, 'w') as f:
                f.write(latex_str)
                f.close()
    except ImportError as e:
        for row_name, row in zip(row_names, table):
            print(row_format.format(row_name, *row))

if __name__ == "__main__":

    dnames = ["brown", "reuters", "gutenberg"]
    datas = []
    unigrams = []
    bigrams = []
    trigrams = []
    # Learn the models for each of the domains, and evaluate it
    for dname in dnames:
        print("-----------------------")
        print(dname)
        data = read_texts("data/corpora.tar.gz", dname)
        datas.append(data)
        # trigram
        from lm import Trigram
        trigram = Trigram()
        trigram.fit_corpus(data.train)
        unigrams.append(set(trigram.vocab()))
        bigrams.append(set(trigram.bigram))
        trigrams.append(set(trigram.trigram))
    
    n = len(dnames)
    overlap_unigram = np.zeros((n,n))
    overlap_bigram = np.zeros((n,n))
    overlap_trigram = np.zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            overlap_unigram[i][j] = len(unigrams[i] & unigrams[j])
            overlap_bigram[i][j] = len(bigrams[i] & bigrams[j])
            overlap_trigram[i][j] = len(trigrams[i] & trigrams[j])

    print("-------------------------------")
    print("unigram")
    print_table(overlap_unigram, dnames, dnames, "table-unigram.tex")
    print("-------------------------------")
    print("bigram")
    print_table(overlap_bigram, dnames, dnames, "table-bigram.tex")
    print("-------------------------------")
    print("trigram")
    print_table(overlap_trigram, dnames, dnames, "table-trigram.tex")