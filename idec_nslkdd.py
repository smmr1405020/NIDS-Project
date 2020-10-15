import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import args
from nslkdd_data_generator import get_training_data, NSLKDD_dataset_test
from sklearn.metrics import confusion_matrix
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(12345)
torch.manual_seed(12345)
import random

random.seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_dataset, labeled_dataset, unlabeled_dataset, normal_dataset = get_training_data(label_ratio=1.0)
test_dataset = NSLKDD_dataset_test()
test_dataset_neg = NSLKDD_dataset_test(test_neg=True)
n_input = total_dataset.get_input_size() + 1

ae_pretrain_epochs = 150
train_dnn_epochs = 200

normal_label = 2


def add_cluster_label(load_cluster_centers_from_numpy=False, load_ds_from_numpy=False):
    data_loader = DataLoader(total_dataset, batch_size=total_dataset.__len__(), shuffle=False)
    clustering = OPTICS(min_samples=5)

    clusters = dict()
    firsttime = 1
    cluster_centers = None
    scaler = StandardScaler()

    if not load_cluster_centers_from_numpy:
        for batch_idx, (x, y, w) in enumerate(data_loader):
            num_data = x.cpu().detach().numpy()
            labels = y.cpu().detach().numpy()
            print("Clustering Started.")
            clustering.fit(num_data)
            print("Clustering ended.")
            cluster_assignment = clustering.labels_

            for i in range(len(cluster_assignment)):
                clusters.setdefault(cluster_assignment[i], []).append(num_data[i])

            cluster_to_label_dict = dict()

            for i in range(len(cluster_assignment)):
                if labels[i] != -1:
                    cluster_to_label_dict.setdefault(cluster_assignment[i], []).append(labels[i])

            label_to_cluster_dict = dict()
            for k, v in cluster_to_label_dict.items():
                if k == -1:
                    continue
                label = np.max(v)
                size = len(v)
                label_to_cluster_dict.setdefault(label, []).append([k, size])

            applicable_clusters = []
            for k, v in label_to_cluster_dict.items():
                total_cluster_selection = min(len(v), 20)
                v = sorted(v, key=lambda item: item[1])
                for i in range(total_cluster_selection):
                    applicable_clusters.append(v[i][0])

            for cluster_id in list(clusters.keys()):
                if cluster_id not in applicable_clusters:
                    del clusters[cluster_id]

            for k, v in clusters.items():
                if k != -1:
                    cluster = np.array(v)
                    kmeans_clustering = KMeans(n_clusters=np.max([int(np.round(cluster.shape[0] / 20)), 1]),
                                               random_state=0)
                    kmeans_clustering.fit(cluster)
                    if firsttime:
                        cluster_centers = np.array(kmeans_clustering.cluster_centers_)
                        firsttime = 0
                    else:
                        cluster_centers = np.concatenate(
                            [cluster_centers, np.array(kmeans_clustering.cluster_centers_)],
                            axis=0)

            np.save(args.cluster_centers_np, cluster_centers)
            break
    else:
        cluster_centers = np.load(args.cluster_centers_np)

    print(cluster_centers.shape[0])

    if not load_ds_from_numpy:
        print("Total Dataset Distance Calculation")
        total_loader = DataLoader(total_dataset, batch_size=total_dataset.__len__(), shuffle=False)
        for batch_idx, (x, y, w) in enumerate(total_loader):
            num_data = x.cpu().detach().numpy()
            firsttime = True
            distances = None
            for i in range(len(num_data)):
                sample = np.expand_dims(num_data[i], axis=0)
                sample_rep = np.repeat(sample, cluster_centers.shape[0], axis=0)
                distance = np.expand_dims(np.sqrt(np.sum((sample_rep - cluster_centers) ** 2, axis=1)), axis=0)

                if not firsttime:
                    distances = np.concatenate([distances, distance], axis=0)
                else:
                    distances = np.array(distance)
                    firsttime = False

            new_x = np.concatenate([num_data, distances], axis=1)

            scaler.fit(new_x)

            new_x = scaler.transform(new_x)
            np.save(args.total_ds_np, new_x)

            total_dataset.set_x(new_x)
            break

        print("Labeled Dataset Distance Calculation")
        labeled_loader = DataLoader(labeled_dataset, batch_size=labeled_dataset.__len__(), shuffle=False)
        for batch_idx, (x, y, w) in enumerate(labeled_loader):
            num_data = x.cpu().detach().numpy()

            firsttime = True
            distances = None
            for i in range(len(num_data)):
                sample = np.expand_dims(num_data[i], axis=0)
                sample_rep = np.repeat(sample, cluster_centers.shape[0], axis=0)
                distance = np.expand_dims(np.sqrt(np.sum((sample_rep - cluster_centers) ** 2, axis=1)), axis=0)

                if not firsttime:
                    distances = np.concatenate([distances, distance], axis=0)
                else:
                    distances = np.array(distance)
                    firsttime = False

            new_x = np.concatenate([num_data, distances], axis=1)
            new_x = scaler.transform(new_x)
            np.save(args.labeled_ds_np, new_x)
            labeled_dataset.set_x(new_x)
            break

        print("Unlabeled Dataset Distance Calculation")
        if unlabeled_dataset.__len__() != 0:
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=unlabeled_dataset.__len__(), shuffle=False)
            for batch_idx, (x, y, w) in enumerate(unlabeled_loader):
                num_data = x.cpu().detach().numpy()
                firsttime = True
                distances = None
                for i in range(len(num_data)):
                    sample = np.expand_dims(num_data[i], axis=0)
                    sample_rep = np.repeat(sample, cluster_centers.shape[0], axis=0)
                    distance = np.expand_dims(np.sqrt(np.sum((sample_rep - cluster_centers) ** 2, axis=1)), axis=0)

                    if not firsttime:
                        distances = np.concatenate([distances, distance], axis=0)
                    else:
                        distances = np.array(distance)
                        firsttime = False

                new_x = np.concatenate([num_data, distances], axis=1)
                new_x = scaler.transform(new_x)
                np.save(args.unlabeled_ds_np, new_x)
                unlabeled_dataset.set_x(new_x)
                break

        print("Normal Dataset Distance Calculation")
        if normal_dataset.__len__() != 0:
            normal_loader = DataLoader(normal_dataset, batch_size=normal_dataset.__len__(), shuffle=False)
            for batch_idx, (x, y, w) in enumerate(normal_loader):
                num_data = x.cpu().detach().numpy()
                firsttime = True
                distances = None
                for i in range(len(num_data)):
                    sample = np.expand_dims(num_data[i], axis=0)
                    sample_rep = np.repeat(sample, cluster_centers.shape[0], axis=0)
                    distance = np.expand_dims(np.sqrt(np.sum((sample_rep - cluster_centers) ** 2, axis=1)), axis=0)

                    if not firsttime:
                        distances = np.concatenate([distances, distance], axis=0)
                    else:
                        distances = np.array(distance)
                        firsttime = False

                new_x = np.concatenate([num_data, distances], axis=1)
                new_x = scaler.transform(new_x)
                np.save(args.normal_ds_np, new_x)
                normal_dataset.set_x(new_x)
                break

        print("Test Dataset Distance Calculation")
        test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False)
        for batch_idx, (x, y, w) in enumerate(test_loader):
            num_data = x.cpu().detach().numpy()

            labels = y.cpu().detach().numpy()

            label, count = np.unique(labels, return_counts=True)
            print(label)
            print(count)

            firsttime = True
            distances = None
            for i in range(len(num_data)):
                sample = np.expand_dims(num_data[i], axis=0)
                sample_rep = np.repeat(sample, cluster_centers.shape[0], axis=0)
                distance = np.expand_dims(np.sqrt(np.sum((sample_rep - cluster_centers) ** 2, axis=1)), axis=0)

                if not firsttime:
                    distances = np.concatenate([distances, distance], axis=0)
                else:
                    distances = np.array(distance)
                    firsttime = False

            new_x = np.concatenate([num_data, distances], axis=1)
            new_x = scaler.transform(new_x)
            np.save(args.test_ds_np, new_x)
            test_dataset.set_x(new_x)
            break

        print("Test Dataset Neg Distance Calculation")
        test_neg_loader = DataLoader(test_dataset_neg, batch_size=test_dataset_neg.__len__(), shuffle=False)
        for batch_idx, (x, y, w) in enumerate(test_neg_loader):
            num_data = x.cpu().detach().numpy()
            labels = y.cpu().detach().numpy()

            label, count = np.unique(labels, return_counts=True)
            print(label)
            print(count)

            firsttime = True
            distances = None
            for i in range(len(num_data)):
                sample = np.expand_dims(num_data[i], axis=0)
                sample_rep = np.repeat(sample, cluster_centers.shape[0], axis=0)
                distance = np.expand_dims(np.sqrt(np.sum((sample_rep - cluster_centers) ** 2, axis=1)), axis=0)

                if not firsttime:
                    distances = np.concatenate([distances, distance], axis=0)
                else:
                    distances = np.array(distance)
                    firsttime = False

            new_x = np.concatenate([num_data, distances], axis=1)
            new_x = scaler.transform(new_x)
            np.save(args.test_ds_neg_np, new_x)
            test_dataset_neg.set_x(new_x)
            break

    else:
        total_dataset.set_x(np.load(args.total_ds_np))
        labeled_dataset.set_x(np.load(args.labeled_ds_np))
        if unlabeled_dataset.__len__() != 0:
            unlabeled_dataset.set_x(np.load(args.unlabeled_ds_np))
        if normal_dataset.__len__() != 0:
            normal_dataset.set_x(np.load(args.normal_ds_np))
        test_dataset.set_x(np.load(args.test_ds_np))
        test_dataset_neg.set_x(np.load(args.test_ds_neg_np))

    print("Done.")


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)

        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


def pretrain_ae(model, data, save_path, epochs):
    '''
    pretrain autoencoder
    '''

    training_data_length = int(0.8 * data.__len__())
    validation_data_length = data.__len__() - training_data_length

    training_data, validation_data = torch.utils.data.random_split(data,
                                                                   [training_data_length, validation_data_length])

    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_ae)

    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)

    min_val_loss = 1000000

    x_dim = data.get_original_feature_size()

    for epoch in range(epochs):
        training_loss = 0.
        validation_loss = 0.
        train_batch_num = 0
        val_batch_num = 0

        model.train()
        for batch_idx, (x, _, idx) in enumerate(train_loader):
            x = x.float()
            x = x.to(device)

            x = x[:, :x_dim]

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

            x = x[:, :x_dim]

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


class NIDS_PREDICTOR(nn.Module):

    def __init__(self, reconstruction_model, normal_model, feature_part_length, cluster_part_length, embedding_size):
        super(NIDS_PREDICTOR, self).__init__()

        self.norm = normal_model
        self.rec = reconstruction_model
        self.feature_part_length = feature_part_length

        self.fc3 = nn.Linear(cluster_part_length + 2 * embedding_size, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 5)

    def forward(self, x):
        x1 = x[:, :self.feature_part_length]
        x2 = x[:, self.feature_part_length:]

        x_bar, z = self.rec(x1)
        x1_bar, z1 = self.norm(x1)

        rec_loss = z - z1

        x_conct = torch.cat([z, x2, rec_loss], dim=-1)
        out_1 = torch.relu(self.fc3(x_conct))
        out_2 = torch.relu(self.fc4(out_1))
        out_3 = self.fc5(out_2)

        return out_3


def train_full_model(load_pretrained_ae=False):
    feature_dimensions = total_dataset.get_original_feature_size()
    cluster_dimensions = total_dataset.get_input_size() - feature_dimensions

    print(feature_dimensions)
    print(cluster_dimensions)

    rec_model = AE(
        n_enc_1=84,
        n_enc_2=63,
        n_enc_3=21,
        n_dec_1=21,
        n_dec_2=63,
        n_dec_3=84,
        n_input=feature_dimensions,
        n_z=args.n_z).to(device)

    norm_model = AE(
        n_enc_1=84,
        n_enc_2=63,
        n_enc_3=21,
        n_dec_1=21,
        n_dec_2=63,
        n_dec_3=84,
        n_input=feature_dimensions,
        n_z=args.n_z).to(device)

    if not load_pretrained_ae:
        pretrain_ae(rec_model, total_dataset, args.rec_model_save_path, ae_pretrain_epochs)
        pretrain_ae(norm_model, normal_dataset, args.norm_model_save_path, ae_pretrain_epochs)

    rec_model.load_state_dict(torch.load(args.rec_model_save_path))
    norm_model.load_state_dict(torch.load(args.norm_model_save_path))

    norm_model.requires_grad_(False)
    # rec_model.requires_grad_(False)

    main_model = NIDS_PREDICTOR(reconstruction_model=rec_model, normal_model=norm_model,
                                feature_part_length=feature_dimensions,
                                cluster_part_length=cluster_dimensions, embedding_size=args.n_z).to(device)

    train_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = Adam([
        # {'params': main_model.rec.parameters(), 'lr' : 0.0001},
        {'params': main_model.fc3.parameters()},
        {'params': main_model.fc4.parameters()},
        {'params': main_model.fc5.parameters()}], lr=0.0009)

    # optimizer = Adam(main_model.parameters(), lr=0.001)
    weights = labeled_dataset.get_weight()
    weights = torch.FloatTensor(weights).to(device)

    min_train_loss = 1000000

    for epoch in range(train_dnn_epochs):
        train_loss = 0.0
        train_batch_num = 0
        train_num_correct = 0
        train_num_examples = 0

        main_model.train()
        for batch_idx, (x, y_t, idx) in enumerate(train_loader):
            x = x.float()
            x = x.to(device)
            train_batch_num = batch_idx

            optimizer.zero_grad()

            y_pred = main_model(x)
            y_t = y_t.to(device)

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
            torch.save(main_model.state_dict(), args.final_model_path)

    print("model saved to {}.".format(args.final_model_path))

    main_model.load_state_dict(torch.load(args.final_model_path))

    main_model.eval()
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False)
    for batch_idx, (x, y_t, idx) in enumerate(test_loader):
        x = x.float()
        x = x.to(device)

        y_pred = main_model(x)
        y_t = y_t.to(device)

        y_pred = torch.max(torch.softmax(y_pred, dim=-1), dim=1)[1]
        y_pred = y_pred.cpu().detach().numpy()
        y_t = y_t.cpu().detach().numpy()

        print(confusion_matrix(y_t, y_pred))

    test_neg_loader = DataLoader(test_dataset_neg, batch_size=test_dataset_neg.__len__(), shuffle=False)
    for batch_idx, (x, y_t, idx) in enumerate(test_neg_loader):
        x = x.float()
        x = x.to(device)

        y_pred = main_model(x)
        y_t = y_t.to(device)

        y_pred = torch.max(torch.softmax(y_pred, dim=-1), dim=1)[1]
        y_pred = y_pred.cpu().detach().numpy()
        y_t = y_t.cpu().detach().numpy()

        print(confusion_matrix(y_t, y_pred))


add_cluster_label(load_cluster_centers_from_numpy=True, load_ds_from_numpy=True)
train_full_model(load_pretrained_ae=True)
