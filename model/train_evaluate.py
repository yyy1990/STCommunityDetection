import joblib
import gensim
import torch
import pandas as pd
from train import train
from evaluate import evaluate

#load pretrain model
check_in = pd.read_csv("./dataset/gowalla/Gowalla_preprocessed_check_in_seq.csv")

#check_in = pd.read_csv('./dataset/brightkite/Brightkite_preprocessed_check_in_seq.csv')

#check_in = pd.read_csv('./dataset/brightkite/Weeplaces_preprocessed_check_in_seq.csv')

# gnn_seq.generate_hyper_graph(check_in=check_in, t_thre=3600, d_thre=10000, topn=50,
# node_save='Weeplaces_node_list.pkl', temporal_save='Weeplaces_temporal_edge.pkl', spatial_save='Weeplaces_spatial_edge.pkl')

label_edge = pd.read_csv("./dataset/gowalla/Gowalla_edges.txt", header=None, sep='\t')  #标注的friend network连边
# label_edge = pd.read_csv("./dataset/brightkite/Brightkite_edges.txt", header=None, sep='\t')  #标注的friend network连边
label_edge = label_edge.rename(columns={0:'userid0', 1:'userid1'})

# label_edge = pd.read_csv('./dataset/weeplaces/weeplace_friends.csv')
# label_edge = label_edge.rename(columns={'userid1':'userid0', 'userid2':'userid1'})

place_id_embedding_model = gensim.models.Word2Vec.load('./result/pretrain_test/Gowalla_placeid_w2v_2.model')
spatial_embedding_model = gensim.models.Word2Vec.load("./result/pretrain_test/Gowalla_spatial_w2v_1.model")
# place_id_embedding_model = gensim.models.Word2Vec.load('./result/brightkite_pretrain/brightkite_placeid.model')
# spatial_embedding_model = gensim.models.Word2Vec.load('./result/brightkite_pretrain/brightkite_spatial.model')
# place_id_embedding_model = gensim.models.Word2Vec.load('./result/weeplaces_pretrain/weeplaces_placeid.model')
# spatial_embedding_model = gensim.models.Word2Vec.load("./result/weeplaces_pretrain/weeplaces_spatial.model")

node_list = joblib.load('gowalla_node_list.pkl')
spatial_edge = joblib.load('gowalla_spatial_edge.pkl')
temporal_edge = joblib.load('gowalla_temporal_edge.pkl')

# node_list = joblib.load('brightkite_node_list.pkl')
# spatial_edge = joblib.load('brightkite_spatial_edge.pkl')
# temporal_edge = joblib.load('brightkite_temporal_edge.pkl')

# node_list = joblib.load('Weeplaces_node_list.pkl')
# spatial_edge = joblib.load('Weeplaces_spatial_edge.pkl')
# temporal_edge = joblib.load('Weeplaces_temporal_edge.pkl')

batch_size=128 #weeplaces 128, 其他256
num_workers=2
select_label = 0.2
input_dim = 640
output_dim = 24
lr = 0.001
save_prefix = './result/gowalla/full/'
print(f'start {save_prefix}')
gpu_id = 0
dropout = 0.1
label_smoothing = 0.05
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
heads = 4
epoch_num = 1
check_point_interval = 1
weighted_remove = True #data aug: weighted remove or random remove
weighted_add = True
spatial_add_prob = 0 #The probability to add from neighbor
topn_select = 3

embedding_selector = 0 #1:只使用基于序列语义信息预训练的embedding, -1: 只使用基于随机游走的预训练, 其他: 两个预训练都用
if embedding_selector == 1:
    input_dim = 128 #gowalla
    #input_dim = 256
elif embedding_selector == -1:
    input_dim = 512 #gowalla
    #input_dim = 384

user_embedding_init = 0 #1: global average, -1: top3: top3 average, 其他: both

aug_num = 1
evaluate_report = save_prefix+'evaluate_report.csv'

#train()
#evaluate()