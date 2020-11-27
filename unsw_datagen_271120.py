from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

np.random.seed(12345)
torch.manual_seed(12345)
import random

random.seed(12345)

binary = False

col_names_unsw = ['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
             'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
             'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
             'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
             'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
             'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
             'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
             'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label']

cat_dict = dict()


def load_unsw_nb15(train_data=True):
    nRowsRead = None  # specify 'None' if want to read whole file

    df1 = pd.read_csv('Dataset_UNSW_NB15/UNSW_NB15_train.csv', delimiter=',',
                      nrows=nRowsRead)
    df1.dataframeName = 'UNSW_NB15_train.csv'

    df2 = pd.read_csv('Dataset_UNSW_NB15/UNSW_NB15_test.csv', delimiter=',')
    df2.dataframeName = 'UNSW_NB15_test.csv'

    # print(df1['attack_cat'].unique())

    df1.sample(frac=1)

    obj_cols = df1.select_dtypes(include=['object']).copy().columns
    obj_cols = list(obj_cols)

    if binary == False:
        df1.drop(['label'], axis=1, inplace=True)
        df2.drop(['label'], axis=1, inplace=True)
        lbl = 'attack_map'
    else:
        df1.drop(['attack_map'], axis=1, inplace=True)
        df2.drop(['attack_map'], axis=1, inplace=True)
        lbl = 'label'

    for col in obj_cols:

        if col != lbl:
            onehot_cols_train = pd.get_dummies(df1[col], prefix=col, dtype='float64')
            onehot_cols_test = pd.get_dummies(df2[col], prefix=col, dtype='float64')

            idx = 0
            for find_col_idx in range(len(list(df1.columns))):
                if list(df1.columns)[find_col_idx] == col:
                    idx = find_col_idx

            itr = 0
            for new_col in list(onehot_cols_train.columns):
                df1.insert(idx + itr + 1, new_col, onehot_cols_train[new_col].values, True)

                if new_col not in list(onehot_cols_test.columns):
                    zero_col = np.zeros(df2.values.shape[0])
                    df2.insert(idx + itr + 1, new_col, zero_col, True)
                else:
                    df2.insert(idx + itr + 1, new_col, onehot_cols_test[new_col].values, True)

                itr += 1

            del df1[col]
            del df2[col]

        else:

            df1[col] = df1[col].astype('category')

            cat_dict_r = dict(enumerate(df1[col].cat.categories))

            for key, value in cat_dict_r.items():
                cat_dict[value] = key

            df1 = df1.replace({col: cat_dict})
            df1[col] = df1[col].astype('int64')

            df2[col] = df2[col].astype('category')
            df2 = df2.replace({col: cat_dict})

            for i in range(len(df2[col])):
                if type(df2[col][i]) is str:
                    df2.at[i, col] = len(cat_dict) + 1

            df2[col] = df2[col].astype('int64')

    train_X = df1.values[:, :-1]
    train_Y = df1.values[:, -1]

    test_X = df2.values[:, :-1]
    test_Y = df2.values[:, -1]

    test_Y = np.array(test_Y).astype(np.int64)

    scaler = StandardScaler()
    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    # print(train_X.shape)
    # print(train_Y.shape)
    # print(test_X.shape)
    # print(test_Y.shape)

    if train_data:
        return train_X, train_Y
    else:
        return test_X, test_Y





#load_unsw_nb15()
