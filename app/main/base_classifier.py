import gc
import os, sys
import time
import jieba
import random
import pickle
import logging
import numpy as np

from gensim import corpora, models

from scipy.sparse import hstack, vstack
from scipy.sparse import csr_matrix

from sklearn.svm import NuSVC
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import time
from functools import wraps
import codecs

from .utility import read_config

logger_info = logging.getLogger('info')
logger_error = logging.getLogger('error')
logger_time = logging.getLogger('time')


# 一个函数装饰器，用来计算函数消耗的时间


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        logger_time.info("Total time running %s : %s seconds" %
                         (function.__name__, str(t1 - t0)))

        return result

    return function_timer


# 向量化类，封装了全部的向量化的操作
class Vectorizer(object):
    dict = None
    tfidf = None
    lsi = None
    keywords = None
    fs = None
    keywords_set = set()
    stopwords_path = None
    dictionary_path = None
    tfidf_path = None
    lsi_path = None
    keywords_path = None
    fs_path = None

    def __init__(
            self,
            n_topics=50,
            max_feature=3000,
            n_job=4
    ):

        self.n_job = n_job
        self.n_topic = n_topics
        self.max_feature = max_feature

        try:
            config = read_config()
            self.stopwords_path = config['vector_path']['stopwords']
            self.dictionary_path = config['vector_path']["dictionary"]
            self.tfidf_path = config['vector_path']["tfidf"]
            self.lsi_path = config['vector_path']["lsi"]
            self.keywords_path = config['vector_path']["keywords"]
            self.fs_path = config['vector_path']["feature_selection"]
            self.idf_path = config['vector_path']["idf_path"]

            dic = config['keywords']
            for key in dic:
                for keyword in key[1].split():
                    self.keywords_set.add(keyword)
                    jieba.add_word(keyword)

        except Exception as e:
            logger_error.error('%s', e)

    def set_paras(self, n_topics=50, max_features=5000):
        self.n_topic = n_topics
        self.max_feature = max_features

    @fn_timer
    def stopwords(self, corpus_raw):
        # jieba.enable_parallel(self.n_job)
        # jieba.add_word("G20")
        with codecs.open(self.stopwords_path, 'r', 'UTF-8') as stopwords:
            stop_words = [line for line in stopwords.read().split('\n')]

        corpus_wl = [[word for word in list(jieba.cut(doc_raw, cut_all=False)) if
                      word not in stop_words and len(word) > 1 and not word.isdigit()] for doc_raw in corpus_raw]

        return corpus_wl

    @fn_timer
    def bagofwords(self, corpus_wl):
        return [self.dict.doc2bow(doc_wl) for doc_wl in corpus_wl]

    @fn_timer
    def build_dictionary(self, corpus_wl):
        dict = corpora.Dictionary(corpus_wl)
        small_freq_ids = [tokenid for tokenid, docfreq in dict.dfs.items() if docfreq < 5]
        dict.filter_tokens(small_freq_ids)
        dict.compactify()
        self.dict = dict
        return dict

    @fn_timer
    def vectorization(self, corpus_raw, mode, flag, enable_csr=True):
        corpus_wl = self.stopwords(corpus_raw)

        if flag == 1:
            if mode < 1:
                self.keywords = corpora.Dictionary.load(self.keywords_path)
                corpus_kw = [self.keywords.doc2bow(doc_wl) for doc_wl in corpus_wl]
                if mode == 0:
                    return self.to_matrix(corpus_kw, (len(corpus_wl), len(self.keywords_set)))

            self.dict = corpora.Dictionary.load(self.dictionary_path)
            self.tfidf = models.TfidfModel.load(self.tfidf_path)

            corpus_tfidf = self.tfidf[self.bagofwords(corpus_wl)]
            if mode == 1:
                return self.to_matrix(corpus_tfidf)

            self.lsi = models.LsiModel.load(self.lsi_path)
            corpus_lsi = self.lsi[corpus_tfidf]
            if mode == 2:
                if not enable_csr:
                    return corpus_lsi
                else:
                    return self.to_matrix(corpus_lsi)

            return self.to_matrix(corpus_tfidf, (len(corpus_wl), len(self.dict.items()))), self.to_matrix(
                corpus_lsi, (len(corpus_wl), self.lsi.num_topics)), self.to_matrix(corpus_kw, (
                len(corpus_wl), len(self.keywords_set)))
        else:
            self.keywords = corpora.Dictionary([list(self.keywords_set)])
            self.keywords.save(self.keywords_path)
            corpus_kw = [self.keywords.doc2bow(doc_wl) for doc_wl in corpus_wl]
            logger_info.info('keywords dictionary train completed %s', self.keywords)

            if mode == 3:
                return self.to_matrix(corpus_kw, (
                    len(corpus_wl), len(self.keywords_set)))

            self.dict = self.build_dictionary(corpus_wl)
            self.dict.save(self.dictionary_path)
            logger_info.info('dictionary train completed %s', self.dict)

            corpus_bow = self.bagofwords(corpus_wl)
            self.tfidf = models.TfidfModel(corpus_bow)
            self.tfidf.save(self.tfidf_path)

            try:
                if len(self.dict.dfs) == len(self.tfidf.idfs):
                    with codecs.open(self.idf_path, "w", encoding="utf-8") as wFile:
                        for i in range(len(self.dict.dfs)):
                            wFile.write(self.dict.get(i).replace('\n', '') + " " + str(self.tfidf.idfs[i]))
                            wFile.write('\r\n')
                else:
                    logger_error.warning("dict and tfidf not euqal")
            except Exception as e:
                logger_error.warning(str(e))

            logger_info.info('tf-idf model train completed %s', self.tfidf)
            corpus_tfidf = self.tfidf[self.bagofwords(corpus_wl)]
            if mode == 1:
                return self.to_matrix(corpus_tfidf)

            self.lsi = models.LsiModel(corpus_tfidf, num_topics=self.n_topic)
            self.lsi.save(self.lsi_path)
            logger_info.info('lsi model train completed %s', self.lsi)
            corpus_lsi = self.lsi[corpus_tfidf]
            if mode == 2:
                return self.to_matrix(corpus_lsi)

            return self.to_matrix(corpus_tfidf, (len(corpus_wl), len(self.dict.items()))), self.to_matrix(
                corpus_lsi, (len(corpus_wl), self.lsi.num_topics)), self.to_matrix(corpus_kw, (
                len(corpus_wl), len(self.keywords_set)))

    @staticmethod
    def to_matrix(corpus_wl, shape=None):
        # 可能需要判断数据的类型来选择转换为数组的方式，目前默认未bag-of-words的方式
        data = []
        rows = []
        cols = []
        line_count = 0
        for line in corpus_wl:
            for elem in line:
                rows.append(line_count)
                cols.append(elem[0])
                data.append(elem[1])
            line_count += 1
        matrix = csr_matrix((data, (rows, cols)), shape=shape)
        return matrix

    def feature_selection(self, features, target, flag):

        if flag == 1:
            self.fs = joblib.load(self.fs_path)
            return self.fs.transform(features)
        else:
            if features.shape[1] > self.max_feature:
                self.fs = SelectKBest(chi2, k='all').fit(features, target)
            else:
                self.fs = SelectKBest(chi2, k='all').fit(features, target)
            logger_info.info('feature selection model train completed %s' % self.fs)
            joblib.dump(self.fs, self.fs_path)
            return self.fs.transform(features)


class Rocchio:
    def __init__(self, alpha=16, beta=4, enable_kmeans=False, enable_lr=True, enable_iterable=False, clusters=5,
                 iteration=10):
        self.alpha = alpha
        self.beta = beta
        self.enable_kmeans = enable_kmeans
        self.enable_lr = enable_lr
        self.enable_iteration = enable_iterable
        self.clusterNum = clusters
        self.iteration = iteration
        self.prototype_plus = None
        self.prototype_minus = None

        try:
            config = read_config()

            self.clf_tfidf_pos_path = config['classifier_path']['tfidf_pos']
            self.clf_lsi_pos_path = config['classifier_path']['lsi_pos']
            self.lr_pos_path = config['classifier_path']['lr_pos']
        except Exception as e:
            logger_error.error('%s', e)

    @fn_timer
    def fit(self, feature_pos, feature_un):
        if feature_pos.shape[0] == 0 or feature_un.shape[0] == 0:
            raise ValueError('invalid value : %s' % 'feature can not be None!')
        if feature_pos.shape[1] != feature_un.shape[1]:
            raise ValueError('invalid value : %s' % 'this function required two matrix with same shape')

        counter_plus = feature_pos.shape[0]
        counter_un = feature_un.shape[0]
        term_plus = np.zeros((1, feature_pos.shape[1]))
        term_minus = np.zeros((1, feature_pos.shape[1]))

        for i in range(counter_plus):
            term_plus += feature_pos[i]

        for i in range(counter_un):
            term_minus += feature_un[i]

        if counter_un != 0 and counter_plus != 0:
            self.prototype_plus = self.alpha * term_plus / counter_plus - self.beta * term_minus / counter_un
            self.prototype_minus = self.alpha * term_minus / counter_un - self.beta * term_plus / counter_plus
            return True

    def cosine_classifier(self, features):
        if self.prototype_plus is None or self.prototype_minus is None:
            raise ValueError('invalid value : %s ', 'prototype is None!')
        if self.prototype_plus.shape[1] != features.shape[1]:
            raise ValueError('invalid value : %s, with size [ %d:%d ] ' % (
                'features and prototype not compatible', self.prototype_plus.shape[1], features.shape[1]))
        return [int(cosine_similarity(self.prototype_plus, features[index]) >= cosine_similarity(
            self.prototype_minus, features[index]))
                for index in range(features.shape[0])]

    @fn_timer
    def extract(self, features_pos, features_un):
        predicts = self.cosine_classifier(features_un)
        rn_indices = [i for i in range(len(predicts)) if predicts[i] != 1]
        if len(rn_indices) == 0:
            raise ValueError('invalid value : %s ' % 'rn is empty')
        if not self.enable_kmeans:
            return rn_indices

        kms = KMeans(n_clusters=self.clusterNum, random_state=0)
        kms.fit(features_un[rn_indices])

        clu_plus = np.zeros((self.clusterNum, features_pos.shape[1]))
        clu_minus = np.zeros((self.clusterNum, features_pos.shape[1]))
        nearest_pos = clu_plus[0]

        for i in range(self.clusterNum):
            term = features_un[[rn_indices[k] for k in range(len(rn_indices)) if kms.labels_[k] == i]]
            troc = Rocchio()
            troc.fit(features_pos, term)
            clu_plus[i], clu_minus[i] = troc.prototype_plus, troc.prototype_minus

        rn_clu_indices = []
        for indice in rn_indices:
            document_matrix = features_un[indice]
            nearest_pos = clu_plus[np.argmax(
                [cosine_similarity(clu_plus[i].reshape(1, -1), document_matrix) for i in
                 range(self.clusterNum)])]
            for i in range(self.clusterNum):
                if cosine_similarity(clu_minus[i].reshape(1, -1), document_matrix) > cosine_similarity(
                        nearest_pos.reshape(1, -1), document_matrix):
                    rn_clu_indices.append(indice)
                    break
        if len(rn_clu_indices) == 0:
            raise ValueError('invalid value : %s ' % 'rn is empty')

        return rn_clu_indices

    @fn_timer
    def build_final_classifier(self, features, target):
        # 函数的参数是训练第一个分类器所需要的正例数据和未标注的数据

        # step0 : check parameters
        if self.enable_lr:
            if len(features) < 2:
                raise ValueError('invalid value %s : ' % 'need two features, tf-idf and lsi')
            else:
                features_tfidf = features[0]
                features_lsi = features[1]

        # step1 : 利用rocchio算法从unlabel数据中提取一些负例,这时候使用tfidf特征
        pos_indices = [i for i in range(len(target)) if target[i] == 1]
        un_indices = [j for j in range(len(target)) if target[j] != 1]
        self.fit(features_tfidf[pos_indices], features_tfidf[un_indices])
        neg_indices = self.extract(features_tfidf[pos_indices], features_tfidf[un_indices])

        # pos, neg = self.extract_RN(features_pos, features_neg)
        # features_neg = features_neg[neg]
        # 训练分类器，可以选择迭代的分类器
        clf_tfidf = self.build_single_svm(features_tfidf[pos_indices], features_tfidf[un_indices], neg_indices, 'tfidf')
        clf_lsi = self.build_single_svm(features_lsi[pos_indices], features_lsi[un_indices], neg_indices, 'lsi')

        # ensemble
        proba_tfidf = clf_tfidf.predict_proba(features_tfidf)
        proba_lsi = clf_lsi.predict_proba(features_lsi)
        predict_tfidf = clf_tfidf.predict(features_tfidf)
        predict_lsi = clf_lsi.predict(features_lsi)

        predict = np.concatenate((predict_tfidf.reshape(-1, 1), predict_lsi.reshape(-1, 1)), axis=1)
        proba = np.concatenate((proba_tfidf, proba_lsi), axis=1)

        lr_pos_indices = [i for i in range(len(pos_indices)) if (predict[i, 0] + predict[i, 1]) > 0]
        if len(neg_indices) > len(lr_pos_indices):
            lr_neg_indices = random.sample(neg_indices, len(lr_pos_indices))
        else:
            lr_neg_indices = neg_indices

        lr_target = np.concatenate((np.ones(len(lr_pos_indices)), -np.ones(len(lr_neg_indices))))

        features_lr = proba[lr_pos_indices + lr_neg_indices]
        lr = LogisticRegression()
        lr.fit(features_lr, lr_target)

        joblib.dump(lr, self.lr_pos_path)
        print("lr score in positive sets : %s " % self.score(lr.predict(proba[:len(pos_indices)]),
                                                             target[:len(pos_indices)]))
        return lr

    @fn_timer
    def build_single_svm(self, feature_pos, feature_un, neg_indices, label):
        if label != 'tfidf' and label != 'lsi':
            return False

        feature_neg = feature_un[neg_indices]
        feature_left = feature_un[[k for k in range(feature_un.shape[0]) if k not in neg_indices]]

        clf = NuSVC(nu=0.1, kernel='linear', probability=True)
        train_feature = vstack((feature_pos, feature_neg))
        train_target = np.concatenate((np.ones(feature_pos.shape[0]), -np.ones(feature_neg.shape[0])))
        clf.fit(train_feature, train_target)
        if label == 'tfidf':
            joblib.dump(clf, self.clf_tfidf_pos_path)
        else:
            joblib.dump(clf, self.clf_lsi_pos_path)

        logger_info.info(
            str(label) + ' score : ' + str(self.score(clf.predict(feature_pos), train_target[:feature_pos.shape[0]])))

        if self.enable_iteration:
            clf_i = NuSVC(nu=0.1, kernel='linear', probability=True)
            for i in range(self.iteration):
                train_feature = vstack((feature_pos, feature_neg))
                train_target = np.concatenate((np.ones(feature_pos.shape[0]), -np.ones(feature_neg.shape[0])))
                clf_i.fit(train_feature, train_target)
                if feature_left.shape[0] == 0:
                    break
                predicts = clf.predict(feature_left)
                n_indices = [item for item in range(len(predicts)) if predicts[item] != 1]
                p_indices = [item for item in range(len(predicts)) if predicts[item] == 1]
                if len(n_indices) > 0:
                    feature_neg = vstack((feature_neg, feature_un[n_indices]))
                    feature_left = feature_left[p_indices]
                else:
                    break
            recall = self.score(clf_i.predict(feature_pos), np.ones(feature_pos.shape[0]))

            logging.info('recall in train sets is : %d' % recall)
            if recall > 0.95:
                return clf_i
        return clf

    def predict(self, features, threshold=0.5):
        # every time load models like tf-idf, lsi.. need to check
        if not self.enable_lr:
            # need to consider the situation about not using lr... to do
            return False

        if len(features) < 2:
            raise ValueError('invalid value : %s' % 'this function required two features')

        clf_tf_idf = joblib.load(self.clf_tfidf_pos_path)
        clf_lsi = joblib.load(self.clf_lsi_pos_path)
        lr = joblib.load(self.lr_pos_path)

        if not isinstance(lr, LogisticRegression):
            raise TypeError('invalid type : %s' % 'logistic regression model load failed ')
        if not isinstance(clf_tf_idf, NuSVC) or not isinstance(clf_lsi, NuSVC):
            raise TypeError('invalid type : %s' % 'classification model load failed ')

        features_tf_idf = features[0]
        features_lsi = features[1]

        if features_tf_idf.shape[1] != clf_tf_idf.shape_fit_[1]:
            raise ValueError('invalid valus : %s' % 'feature shape not compatible with tfi-df model ')
        if features_lsi.shape[1] != clf_lsi.shape_fit_[1]:
            raise ValueError('invalid valus : %s' % 'feature shape not compatible with lsi model ')

        pro_tf_idf = clf_tf_idf.predict_proba(features_tf_idf)
        pro_lsi = clf_lsi.predict_proba(features_lsi)
        probability = lr.predict_proba(np.concatenate((pro_tf_idf, pro_lsi), axis=1))

        results = -np.ones(len(probability))
        for i in range(len(probability)):
            if probability[i, 1] > threshold:
                results[i] = 1
        return results

    @staticmethod
    def score(predict, target, method='recall'):
        if len(predict) != len(target):
            raise ValueError('invalid value : %s ' % 'predict value and target not compatible')
        truth_postive = len([i for i in target if i == 1])
        if method == 'recall' and truth_postive != 0:
            return len([i for i in predict if i == 1]) / truth_postive


if __name__ == "__main__":
    t1 = time.time()

    print(time.time() - t1)
