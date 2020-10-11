from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

np.random.seed(12345)
torch.manual_seed(12345)
import random

random.seed(12345)

# for dirname, _, filenames in os.walk('.\Dataset_NSLKDD'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#
# # .\Dataset_NSLKDD\kdd_test.csv
# # .\Dataset_NSLKDD\kdd_train.csv


ATTACK_DICT = {
    'DoS': ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm',
            'worm'],
    'Probe': ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'],
    'Privilege': ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm'],
    'Access': ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail',
               'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop'],
    'Normal': ['normal']
}

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty_level"]

ATTACK_MAP = dict()
for k, v in ATTACK_DICT.items():
    for att in v:
        ATTACK_MAP[att] = k


def load_nslkdd(train_data=True):

    nRowsRead = 5000  # specify 'None' if want to read whole file

    df1 = pd.read_csv('./Dataset_NSLKDD_2/KDDTrain+.txt', delimiter=',', header=None, names=col_names, nrows=nRowsRead)
    df1.dataframeName = 'KDDTrain+.txt'

    df2 = pd.read_csv('./Dataset_NSLKDD_2/KDDTest+.txt', delimiter=',', header=None, names=col_names)
    df2.dataframeName = 'KDDTest+.txt'

    df1.drop(['difficulty_level'],axis=1, inplace=True)
    df2.drop(['difficulty_level'],axis=1, inplace=True)

    df1.sample(frac=1)

    obj_cols = df1.select_dtypes(include=['object']).copy().columns
    obj_cols = list(obj_cols)

    cat_dict = dict()

    for col in obj_cols:

        if col != 'label':
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

            df1[col] = df1[col].map(ATTACK_MAP)
            df2[col] = df2[col].map(ATTACK_MAP)

            df1[col] = df1[col].astype('category')

            cat_dict_r = dict(enumerate(df1[col].cat.categories))

            for k, v in cat_dict_r.items():
                cat_dict[v] = k

            df1 = df1.replace({col: cat_dict})
            df1[col] = df1[col].astype('int64')

            df2[col] = df2[col].astype('category')
            df2 = df2.replace({col: cat_dict})

            for i in range(len(df2[col])):
                if type(df2[col][i]) is str:
                    df2.at[i, col] = len(cat_dict) + 1

            df2[col] = df2[col].astype('int64')

    # df1 = df1[df1.labels != cat_dict['Normal']]

    train_X = df1.values[:, :-1]
    train_Y = df1.values[:, -1]

    train_Y = np.array(train_Y).astype(np.int64)

    trYunique, trYcounts = np.unique(train_Y, return_counts=True)

    weights = [max(trYcounts) / trYcounts[i] for i in range(len(trYcounts))]
    weights = np.array(weights)

    test_X = df2.values[:, :-1]
    test_Y = df2.values[:, -1]

    test_Y = np.array(test_Y).astype(np.int64)

    scaler = StandardScaler()
    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    if train_data:
        return train_X, train_Y, weights
    else:
        return test_X, test_Y


def get_training_data(label_ratio):
    train_X, train_Y, weights = load_nslkdd(True)
    no_labeled_data = int(label_ratio * len(train_X))

    train_Y[no_labeled_data:] = -1

    labeled_data = train_X[:no_labeled_data], train_Y[:no_labeled_data], weights
    unlabeled_data = train_X[no_labeled_data:], train_Y[no_labeled_data:], weights

    class NSLKDD_dataset_train(Dataset):

        def __init__(self):
            self.labeled_data, self.unlabeled_data = labeled_data, unlabeled_data
            self.x = np.append(self.labeled_data[0], self.unlabeled_data[0], axis=0)
            self.y = np.append(self.labeled_data[1], self.unlabeled_data[1], axis=0)
            self.w = self.labeled_data[2]

        def __len__(self):
            labeled_data_length = self.labeled_data[0].shape[0]
            unlabeled_data_length = self.unlabeled_data[0].shape[0]

            return labeled_data_length + unlabeled_data_length

        def __getitem__(self, idx):
            return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(np.array(self.y[idx])), \
                   torch.LongTensor([idx]).squeeze()

        def get_input_size(self):
            return self.x.shape[1]

        def get_n_clusters(self):
            return len(np.unique(self.y))

        def get_weight(self):
            return self.w
    class NSLKDD_dataset_train_labeled(NSLKDD_dataset_train):

        def __init__(self):
            super(NSLKDD_dataset_train_labeled, self).__init__()
            self.x = self.labeled_data[0]
            self.y = self.labeled_data[1]

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(np.array(self.y[idx])), \
                   torch.LongTensor([idx]).squeeze()
    class NSLKDD_dataset_train_unlabeled(NSLKDD_dataset_train):

        def __init__(self):
            super(NSLKDD_dataset_train_unlabeled, self).__init__()
            self.x = self.unlabeled_data[0]
            self.y = self.unlabeled_data[1]

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(np.array(self.y[idx])), \
                   torch.LongTensor([idx]).squeeze()

    total_dataset = NSLKDD_dataset_train()
    labeled_dataset = NSLKDD_dataset_train_labeled()
    unlabeled_dataset = NSLKDD_dataset_train_unlabeled()

    return total_dataset, labeled_dataset, unlabeled_dataset


class NSLKDD_dataset_test(Dataset):

    def __init__(self):
        self.x, self.y = load_nslkdd(False)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(np.array(self.y[idx])), \
               torch.LongTensor([idx]).squeeze()


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    import scipy.optimize
    ind = scipy.optimize.linear_sum_assignment(np.max(w) - w)

    sm = 0
    for i in range(len(list(ind[0]))):
        x = ind[0][i]
        y = ind[1][i]
        sm += w[x, y]
    return sm * 1.0 / y_pred.size


load_nslkdd(True)