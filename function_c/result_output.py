from sklearn.metrics import classification_report, \
    roc_curve, roc_auc_score, r2_score, f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np

def result_output(y_true, y_pred):

    result = classification_report(y_true, y_pred.round(0), digits=4, output_dict=True)
    print(classification_report(y_true, y_pred.round(0), digits=4))
    p0, r0, f0 = result['0']['precision'], result['0']['recall'], result['0']['f1-score']
    p1, r1, f1 = result['1']['precision'], result['1']['recall'], result['1']['f1-score']
    acc = result['accuracy']
    auc = roc_auc_score(y_true, y_pred)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    ks = max(tpr - fpr)
    print(f'auc: {np.round(auc, 4)}, ks: {np.round(ks, 4)}')

    # print(f"{np.round(auc, 4)}, {np.round(result['1']['recall'], 4)},{np.round(result['1']['precision'], 4)}")

    return acc, auc, ks, p0, r0, f0, p1, r1, f1


def fit_predict(y_true, y_pred, y_c):
    print(classification_report(y_c, y_true,
                                digits=4))
    if y_pred.shape[1] > 2:
        print('AUC value: ', roc_auc_score(y_true, y_pred, multi_class='ovo'))
    if (y_pred.shape[1] > 1) & (y_pred.shape[1] <= 2):
        print('AUC value: ', roc_auc_score(y_true, y_pred[:, 1]))
        fpr, tpr, thresholds = roc_curve(y_true, y_pred[:, 1])
        print('KS value: ', max(tpr - fpr))
        print('R2 score: ', r2_score(y_true=y_true, y_pred=y_pred[:, 1]))


def result_performance(label, y_pred):
    tp = (y_pred[label == 1] >= 0.5).sum()
    fn = (y_pred[label == 1] < 0.5).sum()
    fp = (y_pred[label == 0] >= 0.5).sum()
    correct = (y_pred.round(0) == label).sum() / len(y_pred)
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    print(f'Acc: {np.round(correct, 3)}, '
          f'R: {np.round(recall, 3)}, '
          f'P: {np.round(precision, 3)}, '
          f'f1: {np.round(f1, 3)}')
    return precision, recall, f1, correct


def performance(label, output):
    tp = (output[label == 1] >= 0.5).sum().item()
    fn = (output[label == 1] < 0.5).sum().item()
    fp = (output[label == 0] >= 0.5).sum().item()
    tn = (output[label == 0] <= 0.5).sum().item()
    correct = (output.round() == label).sum().item() / len(label)
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return precision, recall, f1, correct


def performance_tf(label, output):
    tp = len(np.where(output[label == 1] >= 0.5)[0])
    fn = len(np.where(output[label == 1] < 0.5)[0])
    fp = len(np.where(output[label == 0] >= 0.5)[0])

    correct = len(label) - fn - fp
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return precision, recall, f1, correct
