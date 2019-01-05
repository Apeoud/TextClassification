# texrClassification.py

import os
import time
import jieba
import jieba.analyse
import numpy as np
import logging
from gensim import similarities
from .label_classifier import multi_label_classifier, ESConnection
from .base_classifier import Vectorizer
from .base_classifier import Rocchio
import codecs
from .utility import read_config

logger_info = logging.getLogger('info')
logger_error = logging.getLogger('error')
logger_time = logging.getLogger('time')


def load_corpus(load_addr):
    labels = []
    labels_size = []
    corpus = []
    for root, dirs, files in os.walk(load_addr):
        for file in files:
            rFile = open(os.path.join(root, file), 'r', encoding='UTF-8')
            scorpus = [line for line in rFile.readlines()]
            corpus += scorpus
            labels_size.append(len(scorpus))
            labels.append(file.replace(".txt", ""))

    return corpus, labels, labels_size


def train(flag=-1, config_path='/initialization.conf', enable_keywords=False, train_vec=2):
    # step0 : read config

    try:

        config = read_config()
        pos_path = config["datasets_path"]["pos"]
        un_path = config["datasets_path"]["unlabel"]
    except Exception as e:
        logger_error.error("exception occur when config path not found %s " % e)
        return False

    # step1 : load data
    corpora_pos, labels, labels_size = load_corpus(pos_path)
    logger_info.info('length of positive examples in training sets is %d' % len(corpora_pos))
    logger_info.info('distribution of positive examples in training sets is %s' % dict(zip(labels, labels_size)))
    corpora_un, labels_un, labels_size_un = load_corpus(un_path)
    logger_info.info('length of positive examples in unlabeled sets is %d' % len(corpora_un))
    logger_info.info('distribution of unlabeled examples in training sets is %s' % dict(zip(labels_un, labels_size_un)))

    # step1.5 load corpora to es server

    if flag == 1 or flag == -1:
        es = ESConnection()
        labels_es = []
        for i in range(len(labels)):
            for j in range(labels_size[i]):
                labels_es.append(labels[i])
        es.import_data(corpora_pos, labels_es)
        logger_info.info('loading data to es ok!')
        if flag == 1:
            return True

    # step2 : train Dictionary, tf-idf, lsi, features-selection

    vec = Vectorizer()

    vectors = vec.vectorization(corpora_pos + corpora_un, mode=-1, flag=train_vec)
    vectors_fs = vec.feature_selection(vectors[0],
                                       np.concatenate((np.ones(np.sum(labels_size)), -np.ones(np.sum(labels_size_un)))),
                                       train_vec)

    # step3 : train first classifier
    if flag == 2 or flag == -1:
        roc = Rocchio(alpha=16, beta=4, enable_kmeans=True, enable_iterable=True)
        roc.build_final_classifier((vectors_fs, vectors[1]),
                                   np.concatenate((np.ones(np.sum(labels_size)), -np.ones(np.sum(labels_size_un)))))
        logger_info.info('training base classifier ok!')
        if flag == 2:
            return True

    # step4 : load es data and train multi_label classifier
    if flag == 3 or flag == -1:
        mclf = multi_label_classifier(nu=0.2, kernel='linear', enable_keywords=enable_keywords)
        mclf.build_multi_lr(
            (vectors_fs[:np.sum(labels_size)], vectors[1][:np.sum(labels_size)], vectors[2][:np.sum(labels_size)]),
            labels_size, labels)
        logger_info.info('training label classifier ok!')
        logger_info.info('train step finished !!!')
        return True

    return True


def classification(corpora, top=10, base_threshold=0.5, label_threshold=0.8, target=None, store_path='./'):
    # step1: load test sets

    vec = Vectorizer()
    features = vec.vectorization(corpora, mode=-1, flag=1)
    feature_tfidf = vec.feature_selection(features[0], 0, 1)
    feature_lsi = features[1]

    # step2: filter some non-interested documents through base classifier..
    roc = Rocchio()
    results = roc.predict((feature_tfidf, feature_lsi), threshold=base_threshold)
    pos_indices = [i for i in range(len(results)) if results[i] == 1]
    corpora_new = [corpora[pos_indices[i]] for i in range(len(pos_indices))]
    feature_tfidf = feature_tfidf[pos_indices]
    feature_lsi = feature_lsi[pos_indices]

    # step2: get candidate labels, es info add to config , to do..
    es = ESConnection()

    mclf = multi_label_classifier()

    candidate_labels = es.get_candidate_labels(corpora_new, top)
    candidate_score = []
    labels = []
    for i in range(len(candidate_labels)):
        doc_label = []
        doc_score = []
        for label in candidate_labels[i]:
            predict, score = mclf.predict((feature_tfidf[i], feature_lsi[i], features[2][i]), label, 2, label_threshold)
            doc_score.append(score)
            if predict == 1:
                doc_label.append(label)
        candidate_score.append(doc_score)
        labels.append(doc_label)

    if isinstance(store_path, str):
        with codecs.open(store_path, 'w', encoding='utf-8') as wFile:
            for i in range(len(labels)):
                wFile.write(str(labels[i]))
                wFile.write('\r\n')
                wFile.write(corpora[pos_indices[i]])
                wFile.write('\r\n')

    if isinstance(target, list) and len(target) == len(corpora):
        print('base rate : %s' % (len(labels) / len(corpora)))
        hit_num = 0
        for i in range(len(labels)):
            if target[pos_indices[i]] in labels[i]:
                hit_num += 1
        print('hit rate : %s' % (hit_num / len(corpora)))

    return labels


def extract_info(flag, document):
    if flag == 1:
        tags = jieba.analyse.textrank(document, topK=30, withWeight=True)
        return tags

    vec = Vectorizer()
    features = vec.vectorization([document], mode=-1, flag=1)
    feature_tfidf = vec.feature_selection(features[0], 0, 1)
    feature_lsi = features[1]

    if flag == 2:
        roc = Rocchio()
        results = roc.predict((feature_tfidf, feature_lsi), threshold=0.5)
        return results

    es = ESConnection()
    mclf = multi_label_classifier()
    candidate_labels = es.get_candidate_labels([document], 30)
    if flag == 3:
        return candidate_labels[0]

    candidate_score = []
    labels = []
    for i in range(len(candidate_labels)):
        doc_label = []
        doc_score = []
        for j in range(len(candidate_labels[i])):
            label = candidate_labels[i][j]
            predict, score = mclf.predict((feature_tfidf[i], feature_lsi[i], features[2][i]), label, 2, 0.4)
            if predict == 1:
                doc_score.append(score[0])
                doc_label.append(label)

        candidate_score.append(doc_score)
        labels.append([doc_label, doc_score])
    if flag == 4:
        return labels[0]


def sim_lsi(doc_a, doc_b):
    corpus = [doc_a, doc_b]
    vec = Vectorizer()
    feature = vec.vectorization(corpus, mode=2, flag=1, enable_csr=False)
    index = similarities.MatrixSimilarity(feature)
    return index[feature[0]][1]


if __name__ == "__main__":
    begin = time.time()

    train()
    print(time.time() - begin)
