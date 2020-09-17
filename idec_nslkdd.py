import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import args
from nslkdd_data_generator import NSLKDD_dataset_train, cluster_acc

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
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


class IDEC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 alpha=1.0,
                 pretrain_path='data/ae_mnist.pkl'):
        super(IDEC, self).__init__()
        self.alpha = alpha
        self.pretrain_path = pretrain_path

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, path=''):
        # if path == '':
        #     pretrain_ae(self.ae)
        # # load pretrain weights
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained ae from', path)

    def forward(self, x):
        x_bar, z = self.ae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_ae(model):
    '''
    pretrain autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr_ae)
    for epoch in range(100):
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


def train_idec():
    model = IDEC(
        n_enc_1=84,
        n_enc_2=63,
        n_enc_3=21,
        n_dec_1=21,
        n_dec_2=63,
        n_dec_3=84,
        n_input=n_input,
        n_z=args.n_z,
        n_clusters=n_clusters,
        alpha=1.0,
        pretrain_path=args.pretrain_path).to(device)

    #  model.pretrain('data/ae_mnist.pkl')
    model.pretrain()

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # cluster parameter initiate
    data = dataset.x
    y = dataset.y
    data = torch.Tensor(data).to(device)
    x_bar, hidden = model.ae(data)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    nmi_k = nmi_score(y_pred, y)
    print("nmi score={:.4f}".format(nmi_k))

    hidden = None
    x_bar = None

    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model.train()
    for epoch in range(5000):

        if epoch % args.update_interval == 0:

            _, tmp_q = model(data)

            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
            print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),
                  ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

            if epoch > 0 and delta_label < args.tol:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      args.tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        for batch_idx, (x, _, idx) in enumerate(train_loader):

            x = x.float()
            x = x.to(device)
            idx = idx.to(device)

            x_bar, q = model(x)

            reconstr_loss = F.mse_loss(x_bar, x)
            kl_loss = F.kl_div(q.log(), p[idx], reduction='batchmean')
            loss =  reconstr_loss

            # print("r")
            # print(reconstr_loss)
            # print(kl_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


train_idec()