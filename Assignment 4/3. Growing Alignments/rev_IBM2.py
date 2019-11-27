# @Time    : 2019-11-24 14:50
# @Author  : Shihan Ran
# @File    : rev_IBM2.py
# @Software: PyCharm
# @license : Copyright(C), Fudan University
# @Contact : rshcaroline@gmail.com
# @Desc    :


import pickle
from collections import Counter
from itertools import zip_longest as zip


class IBMModel:
    def __init__(self, modelNum, ibm1_path):
        self.modelNum = modelNum
        if modelNum == 2:
            self.ibm1_path = ibm1_path

    def train(self, en_path, es_path, iter_count=5):
        self._train_init(en_path)
        self._em_iterations(en_path, es_path, iter_count)

    def _train_init(self, en_path):
        if self.modelNum == 1:
            self.t = {}
        # For IBM 2, parameter t initialized from IBM 1
        else:
            self.q = {}
            with open(self.ibm1_path, 'rb') as ibm1_file:
                self.t = pickle.load(ibm1_file)
            print('Loaded IBM Model 1.')

        # compute word count in en_file
        en_file = open(en_path)
        self.word_count = Counter()
        for line in en_file:
            tokens = line.rstrip().split()
            tokens = ['_NULL_'] + tokens
            self.word_count += Counter(tokens)

    def _em_iterations(self, en_path, es_path, iter_count):
        for cur_iter in range(iter_count):
            print('iter: ' + str(cur_iter + 1))
            c, c2, c3, c4 = {}, {}, {}, {}
            en_file = open(en_path)
            es_file = open(es_path)
            # iter lines
            for en_line, es_line in zip(en_file, es_file):
                self._em_line(en_line, es_line, c, c2, c3, c4)
            # update t parameter
            for (en, es), en_es_score in c.items():
                self.t[(es, en)] = float(en_es_score) / c2[en]
            if self.modelNum == 2:
                for (j, i, l, m), score in c3.items():
                    self.q[(j, i, l, m)] = float(score) / c4[(i, l, m)]

    def _em_line(self, en_line, es_line, c, c2, c3, c4):
        en_tokens = ['_NULL_'] + en_line.rstrip().split()
        es_tokens = es_line.rstrip().split()
        l, m = len(es_tokens), len(en_tokens)
        for i in range(len(es_tokens)):
            divisor = self._divisor(en_tokens, es_tokens, i)
            for j in range(len(en_tokens)):
                en, es = en_tokens[j], es_tokens[i]
                dividend = self._dividend(en_tokens, es_tokens, i, j)
                delta = float(dividend) / divisor
                c[(en, es)] = c.get((en, es), 0.0) + delta
                c2[en] = c2.get(en, 0) + delta
                if self.modelNum == 2:
                    c3[(j, i, l, m)] = c3.get((j, i, l, m), 0.0) + delta
                    c4[(i, l, m)] = c4.get((i, l, m), 0.0) + delta

    def _dividend(self, en_tokens, es_tokens, i, j):
        en, es = en_tokens[j], es_tokens[i]
        if self.modelNum == 1:
            return self.t.get((es, en), 1.0 / self.word_count[en])
        else:
            l, m = len(es_tokens), len(en_tokens)
            return self.q.get((j, i, l, m), 1.0 / (l + 1)) * self.t.get((es, en), 1.0 / self.word_count[en])

    def parse_sents(self, en_path, es_path, out_path='dev.out'):
        en_file = open(en_path)
        es_file = open(es_path)
        with open(out_path, 'w') as out_file:
            line_num = 1
            for en_line, es_line in zip(en_file, es_file):
                en_line = '_NULL_ ' + en_line
                en_tokens = en_line.rstrip().split()
                es_tokens = es_line.rstrip().split()
                result = self._parse_sent(en_tokens, es_tokens)
                self._print_result(result, out_file, line_num)
                line_num += 1

    def _divisor(self, en_tokens, es_tokens, i):
        base = 0.0
        for j in range(len(en_tokens)):
            en, es = en_tokens[j], es_tokens[i]
            if self.modelNum == 1:
                base += self.t.get((es, en), 1.0 / self.word_count[en])
            else:
                l, m = len(es_tokens), len(en_tokens)
                base += self.t.get((es, en), 1.0 / self.word_count[en]) * self.q.get((j, i, l, m), 1.0 / (l + 1))
        return base

    def _parse_sent(self, en_tokens, es_tokens):
        ans = []
        for i in range(len(es_tokens)):
            max_j = 0
            max_score = 0
            for j in range(len(en_tokens)):
                en, es = en_tokens[j], es_tokens[i]
                score = self.t.get((es, en), 0)
                if self.modelNum == 2:
                    l, m = len(es_tokens), len(en_tokens)
                    score = score * self.q.get((j, i, l, m), 0)

                if score > max_score:
                    max_j, max_score = j, score
            ans.append(max_j)
        return ans

    def _print_result(self, result, out_file, line_num):
        count = 1
        for i in result:
            out_file.write('%d %d %d\r\n' % (line_num, count, i))   # original: i, count
            count += 1


if __name__ == '__main__':
    ibm = IBMModel(2, '../1. IBM Model 1/ibm.model1')
    ibm.train(iter_count=5, en_path='../data/corpus.es', es_path='../data/corpus.en')
    ibm.parse_sents(en_path='../data/dev.es', es_path='../data/dev.en', out_path='rev_alignment.p2.out')
