'''
@Author: 
@Date: 2019-03-29 11:00:03
@LastEditors: Shihan Ran
@LastEditTime: 2019-10-30 10:45:34
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
    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, sentiment)

    # Define a pipeline
    text_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1,3))),
        ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=False, sublinear_tf=True)),
        ('clf', LogisticRegression(random_state=0, C=512, solver='saga', max_iter=1000))
    ])

    ratio = 0.85

    # thresholds = [i/100 for i in range(75, 100)]
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
    ratios = []
    thresholds_train = []
    thresholds_dev = []

    for threshold in thresholds:
        total = list(np.copy(unlabeled.data))     # use copy!
        train = np.copy(sentiment.train_data)
        train_label = np.copy(sentiment.trainy)

        r = []
        # thresholds_train_ = []
        thresholds_dev_ = []

        while(len(total) > (1-ratio)*len(unlabeled.data)):
            print("Total Unlabeled contains: %d, Total Training contains: %d" % (len(total), len(sentiment.train_data)))
            print("\nTraining classifier")
            text_clf.fit(train, train_label)
            
            r.append((len(unlabeled.data)-len(total))/len(unlabeled.data))
            # thresholds_train_.append(classify.evaluate(sentiment.train_data, sentiment.trainy, text_clf, 'train'))
            thresholds_dev_.append(classify.evaluate(sentiment.dev_data, sentiment.devy, text_clf, 'dev'))
            
            print("\nPredicting on unlabeled data")
            yp = text_clf.predict(total)
            prob = text_clf.predict_proba(total)
            itemindex = np.argwhere(prob > threshold)
            indexes = [row for row, col in itemindex]
            
            print("\nAdding labeled data to training data")
            added_data = [total[i] for i in indexes]
            added_label = [yp[i] for i in indexes]      # sentiment.le.transform()
            train = np.concatenate([train, added_data])
            train_label = np.concatenate([train_label, added_label])
            print("\nRemoving labeled data from unlabeled data")
            for index in sorted(indexes, reverse=True):
                del total[index]
        
        ratios.append(r)
        # thresholds_train.append(thresholds_train_)
        thresholds_dev.append(thresholds_dev_)

    for i, threshold in enumerate(thresholds):
        # plt.plot(thresholds, thresholds_train, label='Train')
        plt.plot(ratios[i], thresholds_dev[i], label='threshold = '+str(threshold))
        plt.xlabel('The amount of unlabeled data')
        plt.ylabel('Dev set accuracy')
    
    plt.legend()
    plt.savefig("semi.png")

    # print("\nWriting predictions to a file")
    # write_pred_kaggle_file(unlabeled, text_clf, "data/sentiment-pred.csv", sentiment)
    # write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")

    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
