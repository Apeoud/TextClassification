import os
import numpy as np
from flask import Flask
from flask_script import Manager
from app import create_app

app = create_app()
manager = Manager(app)

from app.main.textClassification import classification, train, load_corpus
from app.main.label_classifier import multi_label_classifier
from app.main.base_classifier import Vectorizer

if __name__ == '__main__':
    manager.run()
    # train(flag=-1, enable_keywords=False, train_vec=2)
    '''
    corpora_test, labels, labels_size = load_corpus('C:/kgtdata/untitled/data/test/positive')
    target = []
    for i in range(len(labels)):
        for j in range(labels_size[i]):
            target.append(labels[i])

    classification(corpora_test, base_threshold=0.5, target=target)
    '''
