import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import args
from nslkdd_data_generator import NSLKDD_dataset_train
from sklearn.metrics import confusion_matrix

np.random.seed(12345)
torch.manual_seed(12345)
import random

random.seed(12345)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = NSLKDD_dataset_train()

n_clusters = dataset.get_input_size()
n_input = dataset.get_input_size()


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


def pretrain_ae(model):
    '''
    pretrain autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr_ae)
    for epoch in range(400):
        total_loss = 0.
        batch_num = 0
        for batch_idx, (x, y, idx) in enumerate(train_loader):
            x = x.float()
            x = x.to(device)
            batch_num = batch_idx

            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_num + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))


def train_autoencoder():
    model = AE(
        n_enc_1=84,
        n_enc_2=63,
        n_enc_3=21,
        n_dec_1=21,
        n_dec_2=63,
        n_dec_3=84,
        n_input=n_input,
        n_z=args.n_z).to(device)

    pretrain_ae(model)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(400):
        total_loss = 0.0
        total_loss_r = 0.0
        total_loss_c = 0.0

        for batch_idx, (x, y_t, idx) in enumerate(train_loader):

            x = x.float()
            x = x.to(device)

            adj_mat = torch.zeros((x.shape[0], x.shape[0])).to(device)
            for i in range(len(adj_mat)):
                for j in range(len(adj_mat[i])):
                    if y_t[i] == y_t[j]:
                        adj_mat[i][j] = 1
                    else:
                        adj_mat[i][j] = 0

            x_bar, z = model(x)

            adj_cap = torch.matmul(z, z.T).to(device)

            reconstr_loss = F.mse_loss(x_bar, x)
            cl_loss = (1.0 / (len(adj_mat) * len(adj_mat))) * F.mse_loss(adj_mat.view(-1), adj_cap.view(-1))

            loss = reconstr_loss + 0.05 * cl_loss

            total_loss += loss.item()
            total_loss_r += reconstr_loss.item()
            total_loss_c += 0.05 * cl_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch {} : Total loss: {:.3f} , Reconstruction Loss: {:.3f} , Cluster Loss: {:.3f}".format(epoch,
                                                                                                          total_loss,
                                                                                                          total_loss_r,
                                                                                                          total_loss_c))

    return model


class NIDS_PREDICTOR(nn.Module):

    def __init__(self, ae_model):
        super(NIDS_PREDICTOR, self).__init__()

        self.fc1 = nn.Linear(args.n_z, 8)
        self.fc2 = nn.Linear(8, 5)
        self.ae = ae_model

    def forward(self, x):
        _, z = self.ae(x)
        out_1 = torch.relu(self.fc1(z))
        out_2 = self.fc2(out_1)

        return out_2


def train_full_model():
    ae_model = train_autoencoder()
    ae_model.requires_grad_(False)

    main_model = NIDS_PREDICTOR(ae_model=ae_model).to(device)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = Adam(main_model.parameters(), lr=0.001)

    weights = dataset.get_weight()
    weights = torch.FloatTensor(weights).to(device)

    for epoch in range(1000):
        total_loss = 0.0
        batch_num = 0

        num_correct = 0
        num_examples = 0

        for batch_idx, (x, y_t, idx) in enumerate(train_loader):
            x = x.float()
            x = x.to(device)
            batch_num = batch_idx

            optimizer.zero_grad()
            y_pred = main_model(x)

            y_t = y_t.to(device)

            loss = torch.nn.CrossEntropyLoss(weight=weights)(y_pred, y_t)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            correct = torch.eq(torch.max(torch.softmax(y_pred,dim=-1), dim=1)[1], y_t).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]

        print("epoch {} loss={:.4f} Accuracy={:.5f}".format(epoch,
                                                            total_loss / (batch_num + 1), num_correct / num_examples))
        torch.save(main_model.state_dict(), args.train_path)
    print("model saved to {}.".format(args.train_path))

    train_loader = DataLoader(dataset, batch_size=4000, shuffle=True)
    for batch_idx, (x, y_t, idx) in enumerate(train_loader):
        x = x.float()
        x = x.to(device)

        y_pred = main_model(x)
        y_t = y_t.to(device)

        y_pred = torch.max(torch.softmax(y_pred,dim=-1), dim=1)[1]
        y_pred = y_pred.cpu().detach().numpy()
        y_t = y_t.cpu().detach().numpy()

        print(confusion_matrix(y_t,y_pred))





train_full_model()


train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


