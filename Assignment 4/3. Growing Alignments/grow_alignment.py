# @Time    : 2019-11-25 12:52
# @Author  : Shihan Ran
# @File    : grow_alignment.py
# @Software: PyCharm
# @license : Copyright(C), Fudan University
# @Contact : rshcaroline@gmail.com
# @Desc    :

import itertools
import numpy


def generate_and_or_file(fname1, fname2):
    with open(fname1) as f:
        alignments1 = [tuple(map(int, line.strip().split(' '))) for line in f]
    groups1 = itertools.groupby(alignments1, key=lambda x:x[0])
    dict1 = dict([(key, list(g)) for key, g in groups1])

    with open(fname2) as f:
        alignments2 = [tuple(map(int, line.strip().split(' '))) for line in f]
    groups2 = itertools.groupby(alignments2, key=lambda x:x[0])
    dict2 = dict([(key, list(g)) for key, g in groups2])

    all_keys = set(dict1.keys()) | set(dict2.keys())
    and_dict = {}
    or_dict = {}
    for key in all_keys:
        aligns1 = set(dict1[key])
        aligns2 = set(dict2[key])
        and_dict[key] = aligns1 & aligns2
        or_dict[key] = aligns1 | aligns2

    and_out = open('and_alignment.out', 'w')
    for key in sorted(and_dict.keys()):
        for item in sorted(and_dict[key], key = lambda x:(x[1], x[2])):
            and_out.write('%s %s %s\n' % item)
    and_out.close()

    or_out = open('or_aligmnent.out', 'w')
    for key in sorted(or_dict.keys()):
        for item in sorted(or_dict[key], key = lambda x:(x[1], x[2])):
            or_out.write('%s %s %s\n' % item)
    or_out.close()
    return


def get_neighbors(e, f, e_length, f_length):
    neighboring = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    neighbors = []
    for neighbor in neighboring:
        _e = e + neighbor[0]
        _f = f + neighbor[1]

        if 1 <= _e <= e_length and 1 <= _f <= f_length:
            neighbors.append((_e, _f))

    return neighbors


def many_to_many_align(fname1, fname2, output_file):
    with open(fname1) as f:
        alignments1 = [tuple(map(int, line.strip().split(' '))) for line in f]
    groups1 = itertools.groupby(alignments1, key=lambda x:x[0])
    result_fe = dict([(key, list(g)) for key, g in groups1])

    with open(fname2) as f:
        alignments2 = [tuple(map(int, line.strip().split(' '))) for line in f]
    groups2 = itertools.groupby(alignments2, key=lambda x:x[0])
    result_ef = dict([(key, list(g)) for key, g in groups2])

    all_keys = set(result_fe.keys()) | set(result_ef.keys())
    and_dict = {}
    or_dict = {}
    for key in all_keys:
        aligns1 = set(result_fe[key])
        aligns2 = set(result_ef[key])
        and_dict[key] = aligns1 & aligns2
        or_dict[key] = aligns1 | aligns2

    sent_count = 200
    with open(output_file, 'w') as out_file:
        for sent_index in range(1, sent_count + 1):
            pairs_fe = [(j-1, k-1) for (i, j, k) in result_fe[sent_index]]
            pairs_ef = [(j-1, k-1) for (i, j, k) in result_ef[sent_index]]

            f_length = max([k for (j, k) in pairs_fe]) + 1
            e_length = max([j for (j, k) in pairs_ef]) + 1

            align_fe = numpy.zeros((e_length + 1, f_length + 1), dtype=numpy.bool)
            align_ef = numpy.zeros((e_length + 1, f_length + 1), dtype=numpy.bool)

            for pair in pairs_fe:
                align_fe[pair] = True
            for pair in pairs_ef:
                align_ef[pair] = True

            alignment = align_fe * align_ef
            union = align_fe + align_ef

            while True:
                has_new_point = False
                for e in range(1, e_length + 1):
                    for f in range(1, f_length + 1):
                        if alignment[e, f]:
                            neighbors = get_neighbors(e, f, e_length, f_length)

                            for neighbor in neighbors:
                                if (not numpy.any(alignment[neighbor[0]]) or not numpy.any(
                                        alignment[:, neighbor[1]])) and union[neighbor]:
                                    alignment[neighbor] = True
                                    has_new_point = True

                if not has_new_point:
                    break

            for e in range(1, e_length + 1):
                for f in range(1, f_length + 1):
                    if alignment[e, f]:
                        out_file.write('{0} {1} {2}\n'.format(sent_index, e+1, f+1))

if __name__ == '__main__':
    generate_and_or_file("alignment.p2.out", "rev_alignment.p2.out")
    many_to_many_align("alignment.p2.out", "rev_alignment.p2.out", "alignment.p3.out")