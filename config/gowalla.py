import os
import torch

result_path = './results/gowalla/'
folder = os.path.exists(result_path)
if not folder:
        os.makedirs(result_path)

#preprocess check-in
origin_check_in = './dataset/gowalla/Gowalla_totalCheckins.txt'
check_in_remove_sparseplace = result_path+'Gowalla_remove_sparse.csv'
sparse_place_thre = 3
check_in_preprocess = result_path+'Gowalla_preprocessed_check_in.csv'

#pretrian semantic
embedding_dim_semantic = 256
pretrain_semantic_path = result_path+'Gowalla_placeid.model'

#pretrian spatial
spatial_walk_num = 10
spatial_walk_len = 512
spatial_walk_potential_nerghbor = 50
embedding_dim_spatial = 384
pretrain_spatial_path = result_path+'Gowalla_spatial.model'

#preprocess global graph
t_thre = 3600
d_thre = 1000
topn = 50
node_path = result_path+'Gowalla_node_list.pkl'
temporal_path = result_path+'Gowalla_temporal_edge.pkl'
spatial_path = result_path+'Gowalla_spatial_edge.pkl'

#train
batch_size = 256
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
label_edge = './dataset/gowalla/Gowalla_edges.txt'
evaluate_report = result_path+'evaluate_report.csv'
