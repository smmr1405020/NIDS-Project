batch_size = 32
lr_ae = 0.0008
lr = 0.0001
n_z = 16
gamma = 0.1
update_interval = 20

pretrain_path = "./Dataset_NSLKDD/ae_nslkdd_d"
train_path = "./Dataset_NSLKDD/main_model_nslkdd_d"

reconstruction_based_ae_pretrain_path = "./Dataset_NSLKDD/reconstruction_ae"
reconstruction_based_ae_pretrain_path_normaldata = "./Dataset_NSLKDD/reconstruction_ae_normaldata"
trained_final_ae_path = "./Dataset_NSLKDD/trained_ae"
final_model_path = "./Dataset_NSLKDD/final_model"


rec_model_save_path = "./Dataset_NSLKDD_2/rec_model"
norm_model_save_path = "./Dataset_NSLKDD_2/norm_model"


cluster_centers_np = "./Dataset_NSLKDD_2/cluster_centers_np_20p.npy"
total_ds_np = "./Dataset_NSLKDD_2/total_ds_np_20p.npy"
normal_ds_np = "./Dataset_NSLKDD_2/normal_ds_np_20p.npy"
labeled_ds_np = "./Dataset_NSLKDD_2/labeled_ds_np_20p.npy"
unlabeled_ds_np = "./Dataset_NSLKDD_2/unlabeled_ds_np_20p.npy"
test_ds_np = "./Dataset_NSLKDD_2/test_ds_np_20p.npy"




tol = 0.0