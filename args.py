from nslkdd_data_generator import NSLKDD_dataset_train

batch_size = 32
lr_ae = 0.001
lr = 0.0001
n_z = 16
gamma = 0.1
update_interval = 20
pretrain_path = "./Dataset_NSLKDD/ae_nslkdd_d"
train_path = "./Dataset_NSLKDD/main_model_nslkdd_d"

tol = 0.0