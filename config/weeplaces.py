import os
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
