import os
import copy
import joblib
import torch
import faiss
import gensim
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from train import index_dataset, graph_dataset
from net import GGT_net
from networkx.algorithms.community import partition_quality


def construct_real_network(check_in, label_edge):
    print('constructing real network...')

    user_set = set(check_in['userid'].unique())
    label_edge['select'] = label_edge.apply(lambda x: True if x['userid0'] in user_set and x['userid1'] in user_set else False, axis=1)
    label_edge = label_edge[label_edge['select']==True]

    G_real = nx.Graph()
    G_real.add_nodes_from(user_set)
    edge_list_construct = [(row['userid0'], row['userid1']) for i, row in label_edge.iterrows()]
    G_real.add_edges_from(edge_list_construct)
    return G_real


@torch.no_grad()
def evaluate_nn(check_in, batch_size, place_id_embedding_model, spatial_embedding_model, spatial_edge, temporal_edge, select_label, topn_select, parameter,
                input_dim, output_dim, heads, dropout, device):
    idx_set = index_dataset(check_in, train=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=idx_set, batch_size=batch_size, shuffle=False, num_workers=8)
    g_set = graph_dataset(check_in, place_id_embedding_model, spatial_embedding_model, spatial_edge, temporal_edge, train=False, select_label=select_label,
    topn=topn_select)

    if torch.cuda.is_available():
        use_gpu = True
        print('evaluating on GPU mode')
    else:
        use_gpu = False
        print('evaluating on CPU mode')

    net = GGT_net(input_dim, output_dim, heads, dropout, device)
    state_dict = torch.load(parameter)

    if use_gpu:
        net = net.to(device=device)
    net.load_state_dict(state_dict)
    net.eval()

    user_list = []
    embedding_list = []

    print('network is inferring...')
    for data in tqdm(test_dataloader):
        users = data.tolist()
        node_embedding, edge_index, edge_attr = g_set[users]
        user_list = user_list+users

        node_embedding = torch.tensor(node_embedding)
        edge_index = torch.tensor(edge_index)
        edge_attr = torch.tensor(edge_attr)
        if use_gpu:
            node_embedding = node_embedding.to(device=device)
            edge_index = edge_index.to(device=device)
            edge_attr = edge_attr.to(device=device)

        number_of_user = len(users)

        x = net(node_embedding, edge_index, edge_attr)
        embedding = x[0:number_of_user]
        embedding_list.append(copy.deepcopy(embedding.cpu().numpy()))
    embedding = np.vstack(embedding_list)

    return embedding, user_list


def evaluate_modularity(array, ncentroids, niter, user_id_list, G_real):
    print('culstering')
    verbose = True
    d = array.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(array)

    print('getting clustering results...')
    D, I = kmeans.index.search(array, k=1)
    community = {}
    for i in range(len(I)):
        c = I[i][0]
        if c not in community:
            community[c] = []
        community[c].append(user_id_list[i])

    community = [community[i] for i in community.keys()]
    print('calculating modularity...')
    modularity = nx.community.modularity(G_real, community)
    coverage, performance = partition_quality(G_real, community)
    return modularity, coverage, performance


def evaluate(check_in, label_edge, save_prefix, evaluate_report_path, batch_size, place_id_embedding_model, spatial_embedding_model, spatial_edge,
             temporal_edge, input_dim, output_dim, heads, device, select_label, topn_select):
    if label_edge.endswith('.txt'):
        label_edge = pd.read_csv(label_edge, header=None, sep='\t')
        label_edge = label_edge.rename(columns={0:'userid0', 1:'userid1'})
    elif label_edge.endswith('.csv'):
        label_edge = pd.read_csv(label_edge)
        label_edge = label_edge.rename(columns={'userid1':'userid0', 'userid2':'userid1'})
    check_in = pd.read_csv(check_in)
    place_id_embedding_model = gensim.models.Word2Vec.load(place_id_embedding_model)
    spatial_embedding_model = gensim.models.Word2Vec.load(spatial_embedding_model)
    spatial_edge = joblib.load(spatial_edge)
    temporal_edge = joblib.load(temporal_edge)
    dropout = 0

    G_real = construct_real_network(check_in, label_edge)
    evaluate_result_list = []

    for f in os.listdir(save_prefix):
        if f.endswith(".pth") and f.startswith('net_parameter'):
            parameter = save_prefix+f
            result_array, _id_array = evaluate_nn(check_in, batch_size, place_id_embedding_model, spatial_embedding_model, spatial_edge, temporal_edge, select_label, topn_select, parameter,
                input_dim, output_dim, heads, dropout, device)

            for ncentroids in [5, 10, 20, 30, 50, 100, 300, 500, 700, 850, 1000]:
                for niter in [30, 40, 50]:
                    modularity, coverage, performance = evaluate_modularity(result_array, ncentroids, niter, _id_array, G_real)
                    ans = {'model': parameter, 'k':ncentroids, 'iter-of-kmeans':niter, 'modularity':modularity, 'coverage': coverage, 'performance': performance}
                    print(ans)
                    evaluate_result_list.append(ans)

    df = pd.DataFrame(evaluate_result_list)
    df.to_csv(evaluate_report_path, index=False)
