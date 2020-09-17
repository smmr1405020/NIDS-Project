from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# for dirname, _, filenames in os.walk('.\Dataset_NSLKDD'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#
# # .\Dataset_NSLKDD\kdd_test.csv
# # .\Dataset_NSLKDD\kdd_train.csv

def load_nslkdd(train_data = True):

    nRowsRead = 1000  # specify 'None' if want to read whole file
    df1 = pd.read_csv('.\Dataset_NSLKDD\kdd_train.csv', delimiter=',', nrows=nRowsRead)
    df1.dataframeName = 'kdd_train.csv'

    df2 = pd.read_csv('.\Dataset_NSLKDD\kdd_test.csv', delimiter=',', nrows=nRowsRead)
    df2.dataframeName = 'kdd_test.csv'

    obj_cols = df1.select_dtypes(include=['object']).copy().columns
    obj_cols = list(obj_cols)

    for col in obj_cols:

        if col != 'labels':
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

            cat_dict = dict()
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

    train_X = df1.values[:, :-1]
    train_Y = df1.values[:, -1]

    test_X = df2.values[:, :-1]
    test_Y = df2.values[:, -1]

    scaler = StandardScaler()
    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    if train_data:
        return train_X , train_Y
    else:
        return test_X, test_Y






class NSLKDD_dataset_train(Dataset):

    def __init__(self):
        self.x, self.y = load_nslkdd(True)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):

        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(np.array(self.y[idx])), \
               torch.LongTensor([idx]).squeeze()

    def get_input_size(self):
        return self.x.shape[1]

    def get_n_clusters(self):
        return len(np.unique(self.y))




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


# dataset = NSLKDD_dataset_train()
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# print(dataset.get_input_size())

# for batch_idx , (x,y,idx) in enumerate(train_loader):
#     print(batch_idx)
#     print(x.shape)
#     print(y)
#     print(idx.shape)
