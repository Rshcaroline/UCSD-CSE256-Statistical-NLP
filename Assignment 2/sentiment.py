'''
@Author: 
@Date: 2019-03-29 11:00:03
@LastEditors: Shihan Ran
@LastEditTime: 2019-10-28 21:44:53
@Email: rshcaroline@gmail.com
@Software: VSCode
@License: Copyright(C), UCSD
@Description: 
'''

#!/bin/python

import numpy as np
import tarfile
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer, HashingVectorizer
from preprocess import clean_text
import classify


def data_analysis(sentiment):
    # Number of characters
    pos_len = [len(s) for i, s in enumerate(sentiment.train_data) if sentiment.train_labels[i] == 'POSITIVE' ]
    neg_len = [len(s) for i, s in enumerate(sentiment.train_data) if sentiment.train_labels[i] == 'NEGATIVE']
    plt.hist(pos_len, rwidth=0.9, align='mid', alpha=0.8, label='Positive Reviews')
    plt.hist(neg_len, rwidth=0.9, align='mid', alpha=0.8, label='Negative Reviews')
    plt.title("Number of Characters")
    plt.legend()
    plt.show()

    # Number of words
    pos_len = [len(s.split(" ")) for i, s in enumerate(sentiment.train_data) if sentiment.train_labels[i] == 'POSITIVE' ]
    neg_len = [len(s.split(" ")) for i, s in enumerate(sentiment.train_data) if sentiment.train_labels[i] == 'NEGATIVE']
    plt.hist(pos_len, rwidth=0.9, align='mid', alpha=0.8, label='Positive Reviews')
    plt.hist(neg_len, rwidth=0.9, align='mid', alpha=0.8, label='Negative Reviews')
    plt.title("Number of Words")
    plt.legend()
    plt.show()

    # Average Word Length
    pos_len = [sum(len(word) for word in s.split(" "))/len(s.split(" ")) for i, s in enumerate(sentiment.train_data) if sentiment.train_labels[i] == 'POSITIVE' ]
    neg_len = [sum(len(word) for word in s.split(" "))/len(s.split(" ")) for i, s in enumerate(sentiment.train_data) if sentiment.train_labels[i] == 'NEGATIVE']
    plt.hist(pos_len, rwidth=0.9, align='mid', bins=range(2,8), alpha=0.8, label='Positive Reviews')
    plt.hist(neg_len, rwidth=0.9, align='mid', bins=range(2,8), alpha=0.8, label='Negative Reviews')
    plt.title("Average Word Length")
    plt.legend()
    plt.show()

    # Number of Stopwords
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    pos_len = [len([x for x in s.split() if x in stop]) for i, s in enumerate(sentiment.train_data) if sentiment.train_labels[i] == 'POSITIVE' ]
    neg_len = [len([x for x in s.split() if x in stop]) for i, s in enumerate(sentiment.train_data) if sentiment.train_labels[i] == 'NEGATIVE']
    plt.hist(pos_len, rwidth=0.9, align='mid', bins=range(2,20), alpha=0.8, label='Positive Reviews')
    plt.hist(neg_len, rwidth=0.9, align='mid', bins=range(2,20), alpha=0.8, label='Negative Reviews')
    plt.title("Number of Stopwords")
    plt.legend()
    plt.show()


def greedy_searchpara(text_clf, sentiment, tarfname):
    # Greedy Search Parameter
    parameters = {
        # 'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (5, 5)],      # (1, 3) is best
        # 'tfidf__use_idf': [(True, False), (True, True), (False, True), ((False, False))],
        'clf__C': [2**(i) for i in range(-10, 15)],     # 512 is best
        # 'clf__class_weight': [None, 'balanced'],  # None is better
        # 'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],    # 'saga' is better
        # 'clf__max_iter': [10**i for i in range(2, 8)],    # iteration 1000
    }
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score
    scoring = {'Accuracy': make_scorer(accuracy_score)}
    gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1, scoring=scoring, refit='Accuracy', return_train_score=True)
    gs_clf = gs_clf.fit(sentiment.train_data, sentiment.trainy)
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    results = gs_clf.cv_results_

    # plotting the result
    plt.figure(figsize=(13, 13))
    # plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
            # fontsize=16)

    plt.xlabel("the inverse of regularization strength for LogisticRegression Model")
    plt.ylabel("Score")

    ax = plt.gca()

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_clf__C'].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

        plt.xscale('log')
        plt.legend(loc="best")
        plt.grid(False)
        plt.show()
    
    # Evaluate on the refit model
    classify.evaluate(sentiment.train_data, sentiment.trainy, gs_clf, 'train')
    classify.evaluate(sentiment.dev_data, sentiment.devy, gs_clf, 'dev')

    # Evaluate on the unlabeled data
    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, sentiment)
    print("Writing predictions to a file")
    write_pred_kaggle_file(unlabeled, gs_clf, "data/sentiment-pred.csv", sentiment)


def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    # print("-- transforming data and labels")
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment


def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
    tar.close()
    return unlabeled


def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels


def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    # yp = cls.predict(unlabeled.X)
    yp = cls.predict(unlabeled.data)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()


def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()


if __name__ == "__main__":
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    print("\nTraining classifier")

    # data_analysis(sentiment)

    # Building a pipeline
    text_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1,3))),
        ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=False, sublinear_tf=True)),
        ('clf', LogisticRegression(random_state=0, C=512, solver='saga', max_iter=1000))
    ])
    text_clf.fit(sentiment.train_data, sentiment.trainy)
    classify.evaluate(sentiment.train_data, sentiment.trainy, text_clf, 'train')
    classify.evaluate(sentiment.dev_data, sentiment.devy, text_clf, 'dev')

    # greedy_searchpara(text_clf, sentiment, tarfname)

    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, sentiment)
    print("Writing predictions to a file")
    write_pred_kaggle_file(unlabeled, text_clf, "data/sentiment-pred.csv", sentiment)
    # write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")

    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
