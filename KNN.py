import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import Linear
from datagen_231120 import get_training_data, dataset_test, dataset_train, cat_dict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import os, glob
import pickle

np.random.seed(12345)
torch.manual_seed(12345)
import random

# 0.01: trainstop 0.005, cluster /25, min_imp_dec: 0.01, (80,100,150) , 5:15PM 26/11/20
# print(cat_dict)

random.seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_dataset, labeled_dataset, unlabeled_dataset = get_training_data(label_ratio=0.01)
test_dataset = dataset_test()
test_dataset_neg = dataset_test(test_neg=True)

ae_epoch = 80
pretrain_epoch = 200
train_epoch = 150

num_data = total_dataset.get_x()
labels = total_dataset.get_y()

total_original_label_counts = dict()
distinct_labels, distinct_label_counts = np.unique(labels, return_counts=True)
for i in range(len(distinct_labels)):
    if distinct_labels[i] != -1:
        total_original_label_counts[distinct_labels[i]] = distinct_label_counts[i]


print(total_original_label_counts)


def tree_work(load_cluster_from_file=False):
    if load_cluster_from_file:
        clustering = pickle.load(file=open('models/clustering.pkl', 'rb'))
    else:
        clustering = KMeans(n_clusters=int(total_dataset.__len__() / 125), random_state=0)
        print("Clustering Started.")
        clustering.fit(num_data)
        print("Clustering ended.")

    cluster_assignment = clustering.labels_
    all_clusters = dict()
    for j in range(len(cluster_assignment)):
        all_clusters.setdefault(cluster_assignment[j], []).append(num_data[j])

    cluster_to_label_dict_ = dict()
    for j in range(len(cluster_assignment)):
        if labels[j] != -1:
            cluster_to_label_dict_.setdefault(cluster_assignment[j], []).append(labels[j])

    for k, v in cluster_to_label_dict_.items():
        cl_label,cl_label_counts = np.unique(np.array(v),return_counts=True)
        desired_label_index = np.argmax(cl_label_counts)
        desired_label = cl_label[desired_label_index]
        cluster_to_label_dict_[k] = desired_label

    return clustering, cluster_to_label_dict_


clustering, cluster_to_label_dict = tree_work(True)
print(cluster_to_label_dict)


def generate_result():
    test_X = test_dataset_neg.get_x()
    test_Y = test_dataset_neg.get_y()

    cluster_assignment = clustering.predict(test_X)

    label = cat_dict['Normal']
    y_pred = np.zeros(test_Y.shape)

    for j in range(len(test_Y)):
        if cluster_assignment[j] in list(cluster_to_label_dict.keys()):
            label = cluster_to_label_dict[cluster_assignment[j]]
        y_pred[j] = label

    print(confusion_matrix(test_Y, y_pred))
    print(classification_report(test_Y, y_pred))

    clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=63)
    clf.fit(dt_X, dt_Y)

generate_result()
