import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from function_c.result_output import *
from xgboost import XGBClassifier

def corr_group(x, y):
    corr = x.join(y).corr()[y.name][x.columns]
    idx = {val: i for i, val in enumerate(np.argsort(corr.values))}
    groups = []
    for l in corr.index:
        if corr[l] > 0:
            groups.append(1)
        else:
            groups.append(0)

    c = Counter(groups)
    groups_num = []
    groups_index = []

    for k in c:
        c_t = [i for i, x in enumerate(groups) if x == k]
        groups_num.append(len(c_t))
        groups_index.append(sorted(c_t, key=lambda x: idx[x]))
    return groups_index,groups_num


def information_gain_group(train_x, test_x, train_y, test_y):

    clf = DecisionTreeClassifier(max_depth=10)
    # clf = XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=50)
    clf.fit(train_x, train_y)
    y_p = clf.predict_proba(test_x)
    result_output(test_y, y_p[:,1])

    q_c = 4
    groups = pd.qcut(clf.feature_importances_,q=q_c,labels=np.arange(q_c).tolist()).codes.tolist()
    idx = {val: i for i, val in enumerate(np.argsort(clf.feature_importances_))}
    c = Counter(groups)
    groups_num = []
    groups_index = []

    for k in c:
        c_t = [i for i, x in enumerate(groups) if x == k]
        groups_num.append(len(c_t))
        groups_index.append(sorted(c_t, key=lambda x: idx[x]))
    return groups_index,groups_num


def feature_group(x, groups_index):
    if type(x) != np.array:
        x = np.float32(x.values)
    x_ = []
    for ll in groups_index:
        x_.append(x[:, ll])
    x = [(matrix,) for matrix in x_]
    x = tuple(zip(*x))
    return x
