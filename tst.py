import numpy as np
import scipy.optimize


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

    print(w)
    print(np.max(w) - w)
    ind = scipy.optimize.linear_sum_assignment(np.max(w) - w)

    print(ind)

    sm = 0
    for i in range(len(list(ind[0]))):
        x = ind[0][i]
        y = ind[1][i]
        sm += w[x, y]
    return sm * 1.0 / y_pred.size


y_true = [0,1,2,1,1,1]
y_pred = [1,2,0,2,2,0]

y_true = np.array(y_true)
y_pred = np.array(y_pred)

ind = scipy.optimize.linear_sum_assignment([[4,1,2],[2,6,1],[3,3,9],[1,2,1],[5,1,1],[6,6,5]])

print(ind)