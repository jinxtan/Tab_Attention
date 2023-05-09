from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def read_nametxt(file_name):
    data = []
    file = open(file_name, 'r')  # open file
    file_data = file.readlines()  # read file
    for row in file_data:
        tmp_list = row.split(': ')  # split data
        tmp_list[-1] = tmp_list[-1].replace('.\n', '')  # delete '\n'
        data.append(tmp_list)  # add data
    data = data[1:len(data) + 1]
    data.append(['target', 'symbolic'])
    return data


def data_read(data_name):
    if data_name == 'Zhongyuan':
        data_ = pd.read_csv('data/Zhongyuan.csv')
        epoch, batch_size = 100, 50
        data_.drop(['Unnamed: 0', 'loan_id', 'user_id'], axis=1, inplace=True)
        target_name = 'isDefault'
        learning_rate = 0.01
        threshold = np.arange(0.45, 1.0, 0.05)

    if 'lc' in data_name:
        data_ = pd.read_csv('data/lendingclub/' + data_name + '.csv').drop(['year'], axis=1)
        data_['issue_d'] = pd.to_datetime(data_['issue_d']).dt.month

        for k in data_.columns:
            if data_[k].dtypes == 'object':
                data_[k] = LabelEncoder().fit_transform(data_[k])
        data_ = data_.sample(frac=1, replace=False, random_state=52, axis=0)
        target_name = 'loan_status'
        learning_rate = 0.01
        threshold = np.arange(0.15, 1.0, 0.05)
        epoch, batch_size = 100, 200

    if data_name == 'statlog':
        data_ = pd.read_csv('data/statlog.csv')
        epoch, batch_size = 300, 30
        target_name = 'Y'
        for i in data_.columns:
            data_[i] = LabelEncoder().fit_transform(data_[i].values)
        learning_rate = 0.01
        threshold = np.arange(0.55, 1.0, 0.05)

    if data_name=='south_german':
        data_ = pd.read_csv('data/SouthGermanCredit.asc', encoding="gbk", engine='python', sep=' ')
        epoch, batch_size = 300, 30
        target_name = 'kredit'
        data_[target_name] = data_[target_name].map({1: 0, 0: 1})

        learning_rate = 0.01
        threshold = np.arange(0.55, 1.0, 0.05)

    if data_name == 'Taiwan':
        data_ = pd.read_excel('data/taiwan.xls', sheet_name='Data', header=0)
        data_.drop(0, axis=0, inplace=True)
        data_.drop(['Unnamed: 0'], axis=1, inplace=True)
        data_ = data_.infer_objects()
        target_name = 'Y'
        epoch, batch_size = 100, 50
        learning_rate = 0.01
        threshold = np.arange(0.50, 1.0, 0.05)


    if 'Unnamed: 0' in data_.columns:
        data_.drop(['Unnamed: 0'], axis=1, inplace=True)

    data = data_.drop([target_name], axis=1, inplace=False)
    targets = data_[target_name] - data_[target_name].min()
    data = pd.DataFrame(MinMaxScaler().fit_transform(data), index=data.index, columns=data.columns)

    return data, targets, learning_rate, epoch, batch_size, threshold, target_name


def fea_sel(clf):
    f_i = pd.DataFrame(data=clf.feature_importances_, index=list(clf.feature_names_in_))
    fea_importance = []
    threshold = 0.001
    for f in f_i.index:
        if f_i.loc[f].abs().values > threshold:
            fea_importance.append(f)
    return fea_importance, f_i

def ci_95(result,sl=3):
    result_ = []
    for col in result.columns:
        print(f"{str(np.round(result[col].mean(), sl))}")
        result_.append(np.round(result[col].mean(), sl))
    for col in result.columns:
        l = str(np.round(result[col].mean(), sl)) + ' ± ' + str(np.round(1.96 * result[col].std() / np.sqrt(len(result[col]) - 1), sl))
        print(l)
        result_.append(l)
    return result_