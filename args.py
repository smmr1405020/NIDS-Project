from nslkdd_data_generator import NSLKDD_dataset_train

batch_size = 32
lr_ae = 0.001
lr = 0.008
n_z = 10
gamma = 0.8
update_interval = 20
pretrain_path = "./Dataset_NSLKDD/ae_nslkdd"
tol = 0.0