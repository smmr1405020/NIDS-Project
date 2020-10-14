import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from torch.nn import Dropout
import args
from nslkdd_data_generator_binary import get_training_data, NSLKDD_dataset_test
from sklearn.metrics import confusion_matrix

np.random.seed(12345)
torch.manual_seed(12345)
import random

random.seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_dataset, labeled_dataset, unlabeled_dataset, normal_dataset = get_training_data(label_ratio=1.0)
n_input = total_dataset.get_input_size()


pretrain_ae_totaldata_epochs = 10
train_ae_epochs = 10
pretrain_ae_normaldata_epochs = 10
train_dnn_epochs = 10


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

    # st_dict = torch.load(save_path)
    # model.load_state_dict(st_dict)


def train_autoencoder(model, load_pretrained_ae=False):
    if not load_pretrained_ae:
        pretrain_ae(model, total_dataset, args.reconstruction_based_ae_pretrain_path, pretrain_ae_totaldata_epochs)

    st_dict = torch.load(args.reconstruction_based_ae_pretrain_path)
    model.load_state_dict(st_dict)

    training_data_length = int(0.8 * labeled_dataset.__len__())
    validation_data_length = labeled_dataset.__len__() - training_data_length

    training_data, validation_data = torch.utils.data.random_split(labeled_dataset,
                                                                   [training_data_length, validation_data_length])

    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=0.0001)

    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)

    x_batch = []
    y_batch = []
    adj_mat_batch = []
    adj_mat_mask_batch = []

    for batch_idx, (x, y_t, _) in enumerate(train_loader):

        x = x.float()
        x = x.to(device)
        x_batch.append(x)

        y_batch.append(y_t)

        adj_mat = torch.zeros((x.shape[0], x.shape[0])).to(device)
        adj_mat_mask = torch.zeros((x.shape[0], x.shape[0])).to(device)

        for i in range(len(adj_mat)):
            for j in range(len(adj_mat[i])):
                if y_t[i] == -1 or y_t[j] == -1:
                    adj_mat[i][j] = 0
                    adj_mat_mask[i][j] = 0
                elif y_t[i] == y_t[j]:
                    adj_mat[i][j] = 1
                    adj_mat_mask[i][j] = 1
                else:
                    adj_mat[i][j] = 0
                    adj_mat_mask[i][j] = 1

        adj_mat_batch.append(adj_mat)
        adj_mat_mask_batch.append(adj_mat_mask)

    x_batch_val = []
    y_batch_val = []
    adj_mat_batch_val = []
    adj_mat_mask_batch_val = []

    for batch_idx, (x, y_t, _) in enumerate(validation_loader):

        x = x.float()
        x = x.to(device)
        x_batch_val.append(x)

        y_batch_val.append(y_t)

        adj_mat_val = torch.zeros((x.shape[0], x.shape[0])).to(device)
        adj_mat_mask_val = torch.zeros((x.shape[0], x.shape[0])).to(device)

        for i in range(len(adj_mat_val)):
            for j in range(len(adj_mat_val[i])):
                if y_t[i] == -1 or y_t[j] == -1:
                    adj_mat_val[i][j] = 0
                    adj_mat_mask_val[i][j] = 0
                elif y_t[i] == y_t[j]:
                    adj_mat_val[i][j] = 1
                    adj_mat_mask_val[i][j] = 1
                else:
                    adj_mat_val[i][j] = 0
                    adj_mat_mask_val[i][j] = 1

        adj_mat_batch_val.append(adj_mat_val)
        adj_mat_mask_batch_val.append(adj_mat_mask_val)

    min_val_loss = 100000

    for epoch in range(train_ae_epochs):

        train_loss = 0.0
        train_loss_r = 0.0
        train_loss_c = 0.0

        validation_loss = 0.0
        validation_loss_r = 0.0
        validation_loss_c = 0.0

        model.train()
        for batch_idx, (x, y_t) in enumerate(zip(x_batch, y_batch)):
            optimizer.zero_grad()
            x = x.float()
            x = x.to(device)

            adj_mat = adj_mat_batch[batch_idx]
            adj_mat_mask = adj_mat_mask_batch[batch_idx]

            x_bar, z = model(x)

            z_n = z / torch.sqrt(torch.sum(z ** 2, dim=1, keepdim=True))
            adj_cap = torch.matmul(z_n, z_n.T).to(device)

            adj_mat = adj_mat * adj_mat_mask
            adj_cap = adj_cap * adj_mat_mask

            reconstr_loss = F.mse_loss(x_bar, x)
            cl_loss = F.mse_loss(adj_mat.view(-1), adj_cap.view(-1))

            loss = reconstr_loss + 0.1 * cl_loss

            train_loss += loss.item()
            train_loss_r += reconstr_loss.item()
            train_loss_c += 0.1 * cl_loss.item()

            loss.backward()
            optimizer.step()

        model.eval()
        for batch_idx, (x, y_t) in enumerate(zip(x_batch_val, y_batch_val)):
            x = x.float()
            x = x.to(device)

            adj_mat = adj_mat_batch_val[batch_idx]
            adj_mat_mask = adj_mat_mask_batch_val[batch_idx]

            x_bar, z = model(x)

            z_n = z / torch.sqrt(torch.sum(z ** 2, dim=1, keepdim=True))
            adj_cap = torch.matmul(z_n, z_n.T).to(device)

            adj_mat = adj_mat * adj_mat_mask
            adj_cap = adj_cap * adj_mat_mask

            reconstr_loss = F.mse_loss(x_bar, x)
            cl_loss = F.mse_loss(adj_mat.view(-1), adj_cap.view(-1))

            loss = reconstr_loss + 0.1 * cl_loss

            validation_loss += loss.item()
            validation_loss_r += reconstr_loss.item()
            validation_loss_c += 0.1 * cl_loss.item()

        if epoch % 1 == 0:
            print("epoch {} : Training Loss: {:.3f},{:.3f},{:.3f} ; Validation Loss:  {:.3f},{:.3f},{:.3f}".
                  format(epoch, train_loss_r, train_loss_c, train_loss, validation_loss_r, validation_loss_c,
                         validation_loss))

        if epoch == 0 or min_val_loss > validation_loss:
            min_val_loss = validation_loss
            torch.save(model.state_dict(), args.trained_final_ae_path)

    print("model saved to {}.".format(args.trained_final_ae_path))

    st_dict = torch.load(args.trained_final_ae_path)
    model.load_state_dict(st_dict)

    return model


class NIDS_PREDICTOR(nn.Module):

    def __init__(self, ae_model, reconstruction_model):
        super(NIDS_PREDICTOR, self).__init__()

        self.fc1 = nn.Linear(args.n_z + 1, 8)
        self.fc2 = nn.Linear(8, 2)
        self.ae = ae_model
        self.rec = reconstruction_model

    def forward(self, x):
        _, z = self.ae(x)
        x_bar, _ = self.rec(x)

        rec_loss = torch.sqrt(torch.sum((x - x_bar) ** 2, dim=1, keepdim=True))

        z_c = torch.cat([z, rec_loss], dim=-1)

        out_1 = torch.relu(self.fc1(z_c))
        out_2 = self.fc2(out_1)

        return out_2


def train_full_model(load_pretrained_ae=False, load_trained_ae=False, load_rec_model=False, not_caring=False):
    ae_model = AE(
        n_enc_1=84,
        n_enc_2=63,
        n_enc_3=21,
        n_dec_1=21,
        n_dec_2=63,
        n_dec_3=84,
        n_input=n_input,
        n_z=args.n_z).to(device)

    if not load_trained_ae:
        train_autoencoder(ae_model, load_pretrained_ae)

    ae_model.load_state_dict(torch.load(args.reconstruction_based_ae_pretrain_path))

    # ae_model.requires_grad_(False)

    rec_model = AE(
        n_enc_1=84,
        n_enc_2=63,
        n_enc_3=21,
        n_dec_1=21,
        n_dec_2=63,
        n_dec_3=84,
        n_input=n_input,
        n_z=args.n_z).to(device)

    if not load_rec_model:
        pretrain_ae(rec_model, normal_dataset, args.reconstruction_based_ae_pretrain_path_normaldata, pretrain_ae_normaldata_epochs)

    rec_model.load_state_dict(torch.load(args.reconstruction_based_ae_pretrain_path_normaldata))

    rec_model.requires_grad_(False)

    if not_caring:
        ae_model = AE(
            n_enc_1=84,
            n_enc_2=63,
            n_enc_3=21,
            n_dec_1=21,
            n_dec_2=63,
            n_dec_3=84,
            n_input=n_input,
            n_z=args.n_z).to(device)

        rec_model = AE(
            n_enc_1=84,
            n_enc_2=63,
            n_enc_3=21,
            n_dec_1=21,
            n_dec_2=63,
            n_dec_3=84,
            n_input=n_input,
            n_z=args.n_z).to(device)

    main_model = NIDS_PREDICTOR(ae_model=ae_model, reconstruction_model=rec_model).to(device)

    training_data_length = int(0.8 * labeled_dataset.__len__())
    validation_data_length = labeled_dataset.__len__() - training_data_length

    training_data, validation_data = torch.utils.data.random_split(labeled_dataset,
                                                                   [training_data_length, validation_data_length])

    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)

    optimizer = Adam([
        {'params': main_model.ae.parameters(), 'lr': 0.001},
        {'params': main_model.fc1.parameters()},
        {'params': main_model.fc2.parameters()}], lr=0.001)

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

    test_loader = DataLoader(NSLKDD_dataset_test(), batch_size=22544, shuffle=False)
    for batch_idx, (x, y_t, idx) in enumerate(test_loader):
        x = x.float()
        x = x.to(device)

        y_pred = main_model(x)
        y_t = y_t.to(device)

        y_pred = torch.max(torch.softmax(y_pred, dim=-1), dim=1)[1]
        y_pred = y_pred.cpu().detach().numpy()
        y_t = y_t.cpu().detach().numpy()

        print(confusion_matrix(y_t, y_pred))


train_full_model(load_pretrained_ae=False,load_trained_ae=False,load_rec_model=False,not_caring=False)
