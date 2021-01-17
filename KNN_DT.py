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

    cluster_to_label_dict = dict()
    for j in range(len(cluster_assignment)):
        if labels[j] != -1:
            cluster_to_label_dict.setdefault(cluster_assignment[j], []).append(labels[j])

    print("Clusters:")
    label_to_cluster_dict = dict()
    for k, v in cluster_to_label_dict.items():
        cl_labels, cl_label_counts = np.unique(np.array(v), return_counts=True)
        print(k)
        print(cl_labels)
        print(cl_label_counts)
        print("\n")
        total_labeled_counts = np.sum(cl_label_counts)

        max_label = np.argmax(cl_label_counts)

        imp_for_label = []
        for label, total_label_count in total_original_label_counts.items():
            for j in range(len(cl_labels)):
                if cl_labels[j] == label and cl_label_counts[j] > 0.1 * total_label_count:
                    imp_for_label.append(label)

        if (cl_label_counts[max_label] / total_labeled_counts) > 0.5:
            selected_label = cl_labels[max_label]
            if len(imp_for_label) == 1:
                if imp_for_label[0] == selected_label:
                    if selected_label != int(cat_dict['Normal']):
                        label = selected_label
                        size = len(v)
                        label_to_cluster_dict.setdefault(label, []).append([k, size])
                    else:
                        if len(cl_labels) == 1:
                            label = selected_label
                            size = len(v)
                            label_to_cluster_dict.setdefault(label, []).append([k, size])
            elif len(imp_for_label) == 0:
                if selected_label != int(cat_dict['Normal']):
                    label = selected_label
                    size = len(v)
                    label_to_cluster_dict.setdefault(label, []).append([k, size])
                else:
                    if len(cl_labels) == 1:
                        label = selected_label
                        size = len(v)
                        label_to_cluster_dict.setdefault(label, []).append([k, size])

    soft_label_mapping = dict()
    for k, v in label_to_cluster_dict.items():
        for cluster_index in v:
            soft_label_mapping[cluster_index[0]] = k

    for j in range(len(labels)):
        if labels[j] == -1 and (int(cluster_assignment[j]) in soft_label_mapping.keys()):
            labels[j] = soft_label_mapping[cluster_assignment[j]]
            labeled_dataset.add_sample(num_data[j], labels[j])

    total_soft_label_counts = dict()
    distinct_slabels, distinct_slabel_counts = np.unique(labels, return_counts=True)
    for j in range(len(distinct_slabels)):
        if distinct_slabels[j] != -1:
            total_soft_label_counts[distinct_slabels[j]] = distinct_slabel_counts[j]

    print(total_soft_label_counts)

    cluster_to_labels_dict = dict()
    for k, v in cluster_to_label_dict.items():
        cl_labels = np.unique(np.array(v))
        cl_labels = sorted(list(cl_labels))
        if cl_labels[0] == -1:
            cl_labels = cl_labels[1:]
        cluster_to_labels_dict[k] = cl_labels

    total_dataset.set_y(labels)

    dt_X = labeled_dataset.get_x()
    dt_Y = labeled_dataset.get_y()

    print(dt_X.shape)
    print(dt_Y.shape)

    print(dt_X.shape)
    print(dt_Y.shape)

    clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=80)
    clf.fit(dt_X, dt_Y)

    print(clf.get_n_leaves())

    file = open('models/tree.pkl', 'wb')
    pickle.dump(clf, file)
    file.close()


tree_work(load_cluster_from_file=True)

def generate_result():
    clf = pickle.load(file=open('models/tree.pkl', 'rb'))

    test_X = test_dataset_neg.get_x()
    test_Y = test_dataset_neg.get_y()

    test_Y_pred = clf.predict(test_X)

    print(confusion_matrix(test_Y, test_Y_pred))
    print(classification_report(test_Y, test_Y_pred))


generate_result()
