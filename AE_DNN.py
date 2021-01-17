import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import Linear
from nslkdd_datagen_231120 import get_training_data, NSLKDD_dataset_test, NSLKDD_dataset_train, cat_dict
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
test_dataset = NSLKDD_dataset_test()
test_dataset_neg = NSLKDD_dataset_test(test_neg=True)

ae_epoch = 80
pretrain_epoch = 80
train_epoch = 150

num_data = total_dataset.get_x()
labels = total_dataset.get_y()

total_original_label_counts = dict()
distinct_labels, distinct_label_counts = np.unique(labels, return_counts=True)
for i in range(len(distinct_labels)):
    if distinct_labels[i] != -1:
        total_original_label_counts[distinct_labels[i]] = distinct_label_counts[i]

print(total_original_label_counts)

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

    print("model saved to {}.".format(save_path))

    return model


pretrain_leaf_dnn('models/pretrain_leaf_dnn', pretrain_epoch)


def generate_result():

    test_X = test_dataset_neg.get_x()
    test_Y = test_dataset_neg.get_y()

    ae_model = AE(total_dataset.get_feature_shape(), 32)
    ae_model.load_state_dict(torch.load('models/train_ae'))
    ae_model.to(device)

    leaf_model = leaf_dnn(32, int(max(labels)) + 1)
    leaf_model.load_state_dict(torch.load('models/pretrain_leaf_dnn'))
    leaf_model.to(device)

    test_Y_pred = np.zeros(test_Y.shape)

    for i in range(len(test_Y)):
        x_emb = ae_model(torch.FloatTensor(test_X[i]).to(device))[1]
        y = torch.softmax(leaf_model(x_emb), dim=-1)
        y_ = y.cpu().detach().numpy()
        y_pred = np.argmax(y_)
        test_Y_pred[i] = y_pred

    print(confusion_matrix(test_Y, test_Y_pred))
    print(classification_report(test_Y, test_Y_pred))



generate_result()