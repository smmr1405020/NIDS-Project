import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import Linear
from nslkdd_datagen_231120 import get_training_data, NSLKDD_dataset_test, NSLKDD_dataset_train, cat_dict
from sklearn.metrics import confusion_matrix
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
total_dataset, labeled_dataset, unlabeled_dataset = get_training_data(label_ratio=1.0)
test_dataset = NSLKDD_dataset_test()
test_dataset_neg = NSLKDD_dataset_test(test_neg=True)

ae_epoch = 80
pretrain_epoch = 100
train_epoch = 150

num_data = total_dataset.get_x()
labels = total_dataset.get_y()

total_original_label_counts = dict()
distinct_labels, distinct_label_counts = np.unique(labels, return_counts=True)
for i in range(len(distinct_labels)):
    if distinct_labels[i] != -1:
        total_original_label_counts[distinct_labels[i]] = distinct_label_counts[i]

print(total_original_label_counts)

def tree_work():
    clustering = KMeans(n_clusters=int(total_dataset.__len__() / 125), random_state=0)
    all_clusters = dict()

    print("Clustering Started.")
    clustering.fit(num_data)
    print("Clustering ended.")
    cluster_assignment = clustering.labels_

    for j in range(len(cluster_assignment)):
        all_clusters.setdefault(cluster_assignment[j], []).append(num_data[j])

    cluster_to_label_dict = dict()
    for j in range(len(cluster_assignment)):
        if labels[j] != -1:
            cluster_to_label_dict.setdefault(cluster_assignment[j], []).append(labels[j])

    label_to_cluster_dict = dict()
    for k, v in cluster_to_label_dict.items():
        cl_labels, cl_label_counts = np.unique(np.array(v), return_counts=True)
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

    dt_X = None
    dt_Y = None
    firsttime = 1

    for j in range(len(num_data)):
        if labels[j] != -1:
            if firsttime == 1:
                dt_X = np.expand_dims(num_data[j], axis=0)
                dt_Y = np.array([labels[j]])
                firsttime = 0
            else:
                dt_X = np.append(dt_X, [num_data[j]], axis=0)
                dt_Y = np.append(dt_Y, [labels[j]])

    print(dt_X.shape)
    print(dt_Y.shape)

    print(dt_X.shape)
    print(dt_Y.shape)

    clf = DecisionTreeClassifier(random_state=0, min_impurity_decrease=0.01)
    clf.fit(dt_X, dt_Y)

    print(clf.get_n_leaves())

    file = open('models/tree.pkl', 'wb')
    pickle.dump(clf, file)
    file.close()

    file = open('models/cluster_to_labels_dict.pkl', 'wb')
    pickle.dump(cluster_to_labels_dict, file)
    file.close()

    file = open('models/clustering.pkl', 'wb')
    pickle.dump(clustering, file)
    file.close()

    leaf_pred = clf.apply(dt_X)

    leaf_dataset_X = dict()
    leaf_dataset_Y = dict()

    for j in range(len(num_data)):
        leaf = clf.apply([num_data[j]])[0]
        if labels[j] != -1:
            leaf_dataset_X.setdefault(leaf, []).append(num_data[j])
            leaf_dataset_Y.setdefault(leaf, []).append(labels[j])

    for k, v in leaf_dataset_X.items():
        leaf_dataset_X[k] = np.array(leaf_dataset_X[k])
        leaf_dataset_Y[k] = np.array(leaf_dataset_Y[k])

    return leaf_dataset_X, leaf_dataset_Y


leaf_dataset_X, leaf_dataset_Y = tree_work()


class AE(nn.Module):

    def __init__(self, n_input, n_z):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, 80)
        self.enc_2 = Linear(80, 50)
        self.z_layer = Linear(50, n_z)

        # decoder
        self.dec_1 = Linear(n_z, 50)
        self.dec_2 = Linear(50, 80)
        self.x_bar_layer = Linear(80, n_input)

    def forward(self, x):
        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        z = self.z_layer(enc_h2)

        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)

        return x_bar, z


def train_ae(epochs, load_from_file=False, save_path='models/train_ae'):
    '''
    train autoencoder
    '''

    model = AE(total_dataset.get_feature_shape(), 32)
    model.to(device)

    ae_train_ds = total_dataset
    training_data_length = int(0.7 * ae_train_ds.__len__())
    validation_data_length = ae_train_ds.__len__() - training_data_length
    training_data, validation_data = torch.utils.data.random_split(ae_train_ds,
                                                                   [training_data_length, validation_data_length])

    train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    min_val_loss = 1000000

    for epoch in range(epochs):
        training_loss = 0.
        validation_loss = 0.
        train_batch_num = 0
        val_batch_num = 0

        model.train()
        for batch_idx, (x, _, idx) in enumerate(train_loader):
            x = x.float()
            x = x.to(device)

            train_batch_num = batch_idx

            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            training_loss += loss.item()

            loss.backward()
            optimizer.step()

        training_loss /= (train_batch_num + 1)

        model.eval()
        for batch_idx, (x, _, idx) in enumerate(validation_loader):
            x = x.float()
            x = x.to(device)

            val_batch_num = batch_idx

            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            validation_loss += loss.item()

        validation_loss /= (val_batch_num + 1)

        if epoch % 1 == 0:
            print(
                "epoch {} , Training loss={:.4f}, Validation loss={:.4f}".format(epoch, training_loss, validation_loss))

        if epoch == 0 or min_val_loss > validation_loss:
            min_val_loss = validation_loss
            torch.save(model.state_dict(), save_path)

    print("model saved to {}.".format(save_path))

    return model


train_ae(ae_epoch, False)


class leaf_dnn(nn.Module):

    def __init__(self, n_input, n_output):
        super(leaf_dnn, self).__init__()
        self.fc1 = nn.Linear(n_input, 8)
        self.fc2 = nn.Linear(8, n_output)

    def forward(self, x):
        out_1 = torch.relu(self.fc1(x))
        out_2 = self.fc2(out_1)

        return out_2


def pretrain_leaf_dnn(save_path, epochs):
    ae_model = AE(total_dataset.get_feature_shape(), 32)
    ae_model.load_state_dict(torch.load('models/train_ae'))
    ae_model.to(device)

    model = leaf_dnn(32, int(max(labels)) + 1)
    model.to(device)

    weights = torch.FloatTensor(labeled_dataset.get_weight()).to(device)

    train_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)  # soft label must be assigned

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    min_train_loss = 1000000
    stop_flag = 1
    prev_train_acc = 0

    for epoch in range(epochs):
        train_loss = 0.0
        train_batch_num = 0
        train_num_correct = 0
        train_num_examples = 0

        model.train()
        for batch_idx, (x, y_t, idx) in enumerate(train_loader):
            x = x.float()
            x = x.to(device)
            train_batch_num = batch_idx

            optimizer.zero_grad()

            x_emb = ae_model(x)[1]
            y_pred = model(x_emb)

            y_t = y_t.clone().detach().to(device)

            loss = torch.nn.CrossEntropyLoss(weight=weights)(y_pred, y_t)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            correct = torch.eq(torch.max(torch.softmax(y_pred, dim=-1), dim=1)[1], y_t).view(-1)
            train_num_correct += torch.sum(correct).item()
            train_num_examples += correct.shape[0]

        train_loss /= (train_batch_num + 1)
        train_acc = train_num_correct / train_num_examples

        if epoch % 1 == 0:
            print("epoch {}; T loss={:.4f} T Accuracy={:.4f}".
                  format(epoch, train_loss, train_num_correct / train_num_examples))

        if epoch == 0 or min_train_loss > train_loss:
            min_train_loss = train_loss
            torch.save(model.state_dict(), save_path)

        if train_acc - prev_train_acc > 0.005:
            stop_flag = 0

        if epoch % 20 == 0:
            if stop_flag == 1:
                break
            stop_flag = 1
            prev_train_acc = train_acc

    print("model saved to {}.".format(save_path))

    return model


pretrain_leaf_dnn('models/pretrain_leaf_dnn', pretrain_epoch)


def train_leaf_dnn(model, dataset, save_path, epochs):
    ae_model = AE(total_dataset.get_feature_shape(), 32)
    ae_model.load_state_dict(torch.load('models/train_ae'))
    ae_model.to(device)

    weights = torch.FloatTensor(total_dataset.get_weight()).to(device)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # soft label must be assigned

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    min_train_loss = 1000000
    prev_train_acc = 0
    stop_flag = 1

    for epoch in range(epochs):
        train_loss = 0.0
        train_batch_num = 0
        train_num_correct = 0
        train_num_examples = 0

        model.train()
        for batch_idx, (x, y_t, idx) in enumerate(train_loader):
            x = x.float()
            x = x.to(device)
            train_batch_num = batch_idx

            optimizer.zero_grad()

            x_emb = ae_model(x)[1]
            y_pred = model(x_emb)

            y_t = y_t.clone().detach().to(device)

            loss = torch.nn.CrossEntropyLoss(weight=weights)(y_pred, y_t)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            correct = torch.eq(torch.max(torch.softmax(y_pred, dim=-1), dim=1)[1], y_t).view(-1)
            train_num_correct += torch.sum(correct).item()
            train_num_examples += correct.shape[0]

        train_loss /= (train_batch_num + 1)
        train_acc = train_num_correct / train_num_examples

        if epoch % 1 == 0:
            print("epoch {}; T loss={:.4f} T Accuracy={:.4f}".
                  format(epoch, train_loss, train_num_correct / train_num_examples))

            if train_acc - prev_train_acc > 0.005:
                stop_flag = 0

        if epoch == 0 or min_train_loss > train_loss:
            min_train_loss = train_loss
            torch.save(model.state_dict(), save_path)

        if train_acc == 1.0:
            break

        if epoch % 20 == 0:
            if stop_flag == 1:
                break
            stop_flag = 1
            prev_train_acc = train_acc

    print("model saved to {}.".format(save_path))

    return model


def create_leaf_dnns():
    filelist = glob.glob(os.path.join('models/leaf_models', "*"))
    for f in filelist:
        os.remove(f)

    for key in leaf_dataset_Y.keys():
        dataset_X = leaf_dataset_X[key]
        dataset_Y = leaf_dataset_Y[key]

        print(key)
        print(dataset_X.shape)
        print(dataset_Y.shape)
        print("\n")

        data = dataset_X, dataset_Y
        dataset = NSLKDD_dataset_train(data)

        model = leaf_dnn(32, int(max(labels)) + 1)
        model.load_state_dict(torch.load('models/pretrain_leaf_dnn'))
        model.to(device)

        save_path = "models/leaf_models/leaf_" + str(key)

        train_leaf_dnn(model, dataset, save_path, train_epoch)


create_leaf_dnns()


def generate_result():
    clf = pickle.load(file=open('models/tree.pkl', 'rb'))
    cluster_to_labels_dict = pickle.load(file=open('models/cluster_to_labels_dict.pkl', 'rb'))
    clustering = pickle.load(file=open('models/clustering.pkl', 'rb'))

    test_X = test_dataset.get_x()
    test_Y = test_dataset.get_y()

    leaf_nodes = clf.apply(test_X)
    cluster_assignment = clustering.predict(test_X)

    ae_model = AE(total_dataset.get_feature_shape(), 32)
    ae_model.load_state_dict(torch.load('models/train_ae'))
    ae_model.to(device)

    model_dict = dict()
    leaf_model_files = os.listdir('models/leaf_models')
    for file in leaf_model_files:
        spl = str(file).split("_")

        leaf_model = leaf_dnn(32, int(max(labels)) + 1)

        if not os.path.exists('models/leaf_models/leaf_' + spl[1]):
            leaf_model.load_state_dict(torch.load('models/pretrain_leaf_dnn'))
        else:
            leaf_model.load_state_dict(torch.load('models/leaf_models/leaf_' + spl[1]))

        leaf_model.to(device)

        model_dict[int(spl[1])] = leaf_model

    test_Y_pred = np.zeros(test_Y.shape)

    for i in range(len(leaf_nodes)):

        leaf_model = model_dict[leaf_nodes[i]]
        x_emb = ae_model(torch.FloatTensor(test_X[i]).to(device))[1]
        y = torch.softmax(leaf_model(x_emb), dim=-1)
        y_ = y.cpu().detach().numpy()
        if int(cluster_assignment[i]) in cluster_to_labels_dict.keys():
            allowable_labels = cluster_to_labels_dict[int(cluster_assignment[i])]
            if len(allowable_labels) != 1 or allowable_labels[0] != int(cat_dict['Normal']):
                mask = np.zeros(int(max(labels)) + 1)
                for j in range(len(allowable_labels)):
                    mask[int(allowable_labels[j])] = 1
                y_pred = np.argmax(y_ * mask)
            else:
                y_pred = np.argmax(y_)
        else:
            y_pred = np.argmax(y_)

        test_Y_pred[i] = y_pred

    print(confusion_matrix(test_Y, test_Y_pred))


generate_result()
