from collections import Counter

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, r2_score
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


def multi_view_Kmeans(train_x, k):
    '''
    :param train_x: the data need to be clustered
    :param k: the number of cluster
    :return: the cluster result
    '''
    clf = KMeans(n_clusters=k)
    clf.fit(np.transpose(train_x))
    groups = clf.predict(np.transpose(train_x))

    c = Counter(groups)
    groups_num = []
    groups_index = []

    for k in c:
        c_t = [i for i, x in enumerate(groups) if x == k]
        groups_num.append(len(c_t))
        groups_index.append(c_t)
    return groups_index, groups_num

