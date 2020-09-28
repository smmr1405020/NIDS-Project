import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import args
from nslkdd_data_generator import NSLKDD_dataset_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = NSLKDD_dataset_train()

n_clusters = dataset.get_input_size()
n_input = dataset.get_input_size()
import pprint



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
        enc_h1 = F.tanh(self.enc_1(x))
        enc_h2 = F.tanh(self.enc_2(enc_h1))
        enc_h3 = F.tanh(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        # decoder
        dec_h1 = F.tanh(self.dec_1(z))
        dec_h2 = F.tanh(self.dec_2(dec_h1))
        dec_h3 = F.tanh(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


def pretrain_ae(model):
    '''
    pretrain autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr_ae)
    for epoch in range(50):
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


def test_reconstruction_error():
    model = AE(84, 63, 42, 42, 63, 84, n_input, 16).to(device)
    pretrain_ae(model)

    model_st_dict = torch.load(args.pretrain_path)
    model.load_state_dict(model_st_dict)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_labels = []
    all_label_losses = []
    et_count = []

    for i in range(100):
        all_labels.append(i)
        all_label_losses.append([])
        et_count.append(0)

    for (x, y, idx) in train_loader:
        x = x.float()
        x = x.to(device)

        x_bar, z = model(x)

        loss = F.mse_loss(x_bar, x)
        loss_it = loss.item()

        current_label = int(y.item())
        if loss_it < 30.0:
            all_label_losses[current_label].append(loss_it)
        else:
            et_count[current_label] += 1

    # all_label_counts = []
    # all_label_loss_mean = []
    # all_label_loss_std = []
    # all_label_loss_min = []
    # all_label_loss_max = []

    ls = []
    for i in range(len(all_label_losses)):
        if len(all_label_losses[i]) > 0:
            count = len(all_label_losses[i])
            mean = np.mean(all_label_losses[i])
            std = np.std(all_label_losses[i])
            min = np.min(all_label_losses[i])
            max = np.max(all_label_losses[i])
            ls.append((i, count, et_count[i], mean, std, min, max))

    ls = sorted(ls, key=lambda item: item[1], reverse=True)

    pprint.PrettyPrinter(indent=4).pprint(ls)


test_reconstruction_error()
