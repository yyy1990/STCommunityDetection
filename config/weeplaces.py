import os
import torch

result_path = './results/weeplaces/'
folder = os.path.exists(result_path)
if not folder:
        os.makedirs(result_path)

#preprocessing config
origin_check_in = './dataset/weeplaces/weeplace_checkins.csv'
check_in_remove_sparseplace = result_path+'Weeplaces_remove_sparse.csv'
sparse_place_thre = 3
check_in_preprocess = result_path+'Weeplaces_preprocessed_check_in.csv'

#pretrian semantic
embedding_dim_semantic = 256
pretrain_semantic_path = result_path+'Weeplaces_placeid.model'

#pretrian spatial
spatial_walk_num = 10
spatial_walk_len = 512
spatial_walk_potential_nerghbor = 50
embedding_dim_spatial = 384
pretrain_spatial_path = result_path+'Weeplaces_spatial.model'

#preprocess global graph
t_thre = 3600
d_thre = 1000
topn = 50
node_path = result_path+'Weeplaces_node_list.pkl'
temporal_path = result_path+'Weeplaces_temporal_edge.pkl'
spatial_path = result_path+'Weeplaces_spatial_edge.pkl'

#train
batch_size = 128
input_dim = 640
output_dim_head = 24
heads = 4
dropout = 0.1
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
lr = 0.001
label_smoothing = 0.05
epoch_num = 20
check_point_interval = 1

#evaluate
label_edge = './dataset/weeplaces/weeplace_friends.csv'
evaluate_report = result_path+'evaluate_report.csv'
