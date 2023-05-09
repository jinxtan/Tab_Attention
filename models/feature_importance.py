import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans


def feature_importance(train_x, train_y):
    '''
    :param train_x: the data need to be clustered
    :param k: the number of cluster
    :return: the cluster result
    '''

    index_n = ['RF', 'LR', 'DT', 'XGB','AdaBoost','GBT', 'K-means', 'corr']

    feature_importance = pd.DataFrame(index=train_x.columns, columns=index_n)
    model_l = [RandomForestClassifier(n_estimators=50, max_depth=10, random_state=10),
               LogisticRegression(),
               DecisionTreeClassifier(max_depth=10, random_state=10),
               XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=50, random_state=10),
               AdaBoostClassifier(n_estimators=50, random_state=10),
               GradientBoostingClassifier(n_estimators=50, max_depth=10, random_state=10),
               KMeans(n_clusters=4)
               ]
    corr = train_x.join(train_y).corr().iloc[:-1, -1]
    feature_importance.loc[:,'corr'] = corr

    for l in range(len(model_l)):
        rf = model_l[l]
        if index_n[l] == 'K-means':
            rf.fit(np.transpose(train_x))
        else:
            rf_model = rf.fit(train_x, train_y)
        if index_n[l] == 'LR':
            f_i = rf_model.coef_[0, :]
        elif index_n[l] == 'K-means':
            f_i = rf.labels_
        else:
            f_i = rf_model.feature_importances_

        feature_importance.loc[:,index_n[l]] = f_i

    return feature_importance


def feature_select(f_importance, threshold=0.05, f_i_num=1):
    fea_importance = []
    for f_i in f_importance.columns:
        l = np.where(f_importance[f_i].abs() > threshold)
        if l[0].shape[0] > f_i_num:
            fea_importance.append(f_i)
    return fea_importance

def col_select(clf, train_x,test_x,train_y,threshold = 0.01, auto=False,operator='<='):
    if clf == 'corr':
        corr = train_x.join(train_y).corr().iloc[:-1,-1].sort_values()
        if auto:
            threshold = np.quantile(abs(corr), 0.7)

        index = eval('abs(corr)'+operator+'threshold')

    elif clf == 'kmeans':
        clf = KMeans(n_clusters=4)
        clf.fit(np.transpose(train_x))
        c = Counter(clf.labels_)
        index = clf.labels_==c.most_common(1)[0][0]

    elif clf == 'lr':
        clf = LogisticRegression()
        clf.fit(train_x,train_y)

        feature_i = clf.coef_[0]
        if auto:
            threshold = np.quantile(abs(feature_i), 0.7)
        index = eval('abs(feature_i)'+operator+'threshold')

    else:
        clf.fit(train_x,train_y)
        feature_i = clf.feature_importances_
        if auto:
            threshold = np.quantile(feature_i, 0.7)
        index = eval('feature_i'+operator+'threshold')
    print(threshold)
    train_x = train_x.loc[:, index]
    test_x = test_x.loc[:, index]
    return train_x,test_x

def col_group(clf, train_x,train_y,threshold = 0.01, auto=False,operator='>'):
    if clf == 'corr':
        corr = train_x.join(train_y).corr().iloc[:-1,-1]
        corr = corr.drop(corr[corr.isna()].index,axis=0)
        if auto:
            threshold = np.sort(abs(corr))[int(len(corr)*0.7)]
        index = eval('abs(corr)'+operator+'threshold')

    elif clf == 'kmeans':
        clf = KMeans(n_clusters=4)
        clf.fit(np.transpose(train_x))
        c = Counter(clf.labels_)
        index = clf.labels_==c.most_common(1)[0][0]

    elif clf == 'lr':
        clf = LogisticRegression()
        clf.fit(train_x,train_y)

        feature_i = clf.coef_[0]
        if auto:
            threshold = np.sort(abs(feature_i))[int(len(feature_i)*0.7)]
        index = eval('abs(feature_i)'+operator+'threshold')

    else:
        clf.fit(train_x,train_y)
        feature_i = clf.feature_importances_
        if auto:
            threshold = np.sort(feature_i)[int(len(feature_i)*0.7)]
        index = eval('feature_i'+operator+'threshold')
    print(str(clf)[:10],threshold)

    return index

def feature_group_multi_way(train_x, train_y,view_num=6):
    '''
    :param train_x: the data need to be clustered
    :param k: the number of cluster
    :return: the cluster result
    '''
    clfs = [DecisionTreeClassifier(max_depth=10, random_state=10),
            GradientBoostingClassifier(n_estimators=50, max_depth=10, random_state=10),
            RandomForestClassifier(n_estimators=50, max_depth=10, random_state=10),
            'lr', 'kmeans', 'corr']
    clfs = clfs[:view_num]
    groups_index = []
    groups_num = []
    for clf in clfs:
        index = col_group(clf, train_x,train_y,threshold = 0.01, auto=True,operator='>')
        groups_num.append(np.count_nonzero(index))
        groups_index.append(np.where(index)[0])

    return groups_index, groups_num