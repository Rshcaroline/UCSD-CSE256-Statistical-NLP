'''
@Author: 
@Date: 2017-03-12 04:05:36
@LastEditors: Shihan Ran
@LastEditTime: 2019-10-09 13:58:25
@Email: rshcaroline@gmail.com
@Software: VSCode
@License: Copyright(C), UCSD
@Description: The interface and a simple implementation of a language model.
'''

#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences.
        """
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))   # perplexity = 2^{entropy}

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1   # + 1 for EOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])   # wi & w0, w1, ... w{i-1}
        p += self.cond_logprob('END_OF_SENTENCE', sentence)     # for EOS
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): 
        pass

    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): 
        pass
    
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): 
        pass
    
    # required, the list of words the language model suports (including EOS)
    def vocab(self): 
        pass


class Unigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.lbackoff = log(backoff, 2)

    def inc_word(self, w):
        """Count the appearance"""
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        """Update the model when a sentence is observed"""
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot  # normalize: loga-logb = log(a/b)

    def cond_logprob(self, word, previous):
        if word in self.model:
            return self.model[word]     # unigram: doesn't depend on previous words
        else:
            return self.lbackoff

    def vocab(self):
        return self.model.keys()

