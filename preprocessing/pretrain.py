import time
from tqdm import tqdm
import gensim
import pandas as pd
from transbigdata import getdistance
import math
import random
import copy
from sklearn.neighbors import KDTree
import numpy as np


def pretrain_embedding(check_in, sort_col='timestamp', id_col='userid', target_col='placeid', embedding_dim=128, save_path='Gowalla_placeid_w2v.model',
sg=1, min_count=1, workers=10, window=5, epochs=10, split = True, min_split = 10, data_aug = 2, real_sequence=True):
    check_in = check_in.sort_values(sort_col)
    userid_list = list(check_in[id_col].unique())
    corpus = []
    col = target_col
    if type(col)==list:
        check_in['word'] = ''
        for c in col:
            check_in['word'] = check_in['word']+'-'+check_in[c].astype(str)
    else:
        check_in['word'] = check_in[col]
    check_in_group = check_in.groupby(id_col)
    print('getting corpus...')
    if real_sequence:
        for userid in tqdm(userid_list):
            group = check_in_group.get_group(userid)
            trajectory = list(group['word'])
            corpus.append(trajectory)
            if split and len(trajectory)>=min_split:
                for i in range(data_aug):
                    ind = random.randint(2, len(trajectory)-2)
                    tra1 = trajectory[0:ind]
                    tra2 = trajectory[ind+1::]
                    tra3 = tra1+tra2
                    corpus.append(tra1)
                    corpus.append(tra2)
                    corpus.append(tra3)
    else:
        tokens = sorted(list(check_in[target_col].unique()))
        for i in range(len(tokens)):
            if i == 0:
                tra = tokens
            else:
                tra = tokens[i::]+tokens[0:i]
            corpus.append(tra)
            corpus.append(tra.reverse())
    t0 = time.time()
    print(f'corpus size:{len(corpus)}')
    print(f'pretraining {col}...')
    model = gensim.models.Word2Vec(corpus, min_count=min_count, workers=workers, window=window, vector_size=embedding_dim, epochs=epochs, sg=sg)
    model.save(save_path)
    t1 = time.time()
    print(f'pretraining {col} complete, {t1-t0} seconds, dict size={len(model.wv.key_to_index)}, corpus size={len(corpus)}')
    return model


def pretrain_spatial_embedding(origin_check_in="./dataset/gowalla/Gowalla_totalCheckins.txt", walk_num=10, walk_len=128, potential_nerghbor=50, embedding_dim=374,
save_path='Gowalla_spatial_w2v.model', window=5, epochs=10, sg=1):
    if origin_check_in.endswith('txt'):
        df = pd.read_csv(origin_check_in, header=None, sep='\t')
        df = df.rename(columns={0:'userid', 1:'time', 2:'lat', 3:'lon', 4:'placeid'})
    else:
        df = pd.read_csv(origin_check_in)
    placeid_df = df.groupby('placeid').head(1)
    placeid_df = placeid_df[['placeid', 'lat', 'lon']]
    distance_dict = {}
    placeid_list = list(placeid_df['placeid'].unique())
    placeid_df.set_index(keys='placeid', inplace=True)

    print('getting coordinates...')
    coord = []
    for id1 in tqdm(placeid_list):
        lat = placeid_df.loc[id1]['lat']
        lon = placeid_df.loc[id1]['lon']
        coord.append([lat, lon])
    coord = np.array(coord)
    print('constructing KD tree ...')
    tree = KDTree(coord, leaf_size=potential_nerghbor)
    print('KD tree constructed')
    print('getting neighbor distances...')
    for id1 in tqdm(placeid_list):
        distance_dict[id1] = {}
        lat1 = placeid_df.loc[id1]['lat']
        lon1 = placeid_df.loc[id1]['lon']
        dist, ind = tree.query(np.array([lat, lon]).reshape(1, -1), k=potential_nerghbor)
        for i in range(len(ind[0])):
            id2 = placeid_list[ind[0][i]]
            if id2!=id1:
                distance_dict[id1][id2] = distance2weight(dist[0][i])

    print('generateing random walk paths...')
    corpus = []
    d = {'distance':[], 'id':[], 'num':[]}
    for id1 in tqdm(placeid_list):
        for i in range(walk_num):
            current = id1
            walk_step = 0
            trajectory = [copy.deepcopy(current)]
            while walk_step<walk_len:
                current = random.choices(population=list(distance_dict[current].keys()), weights=list(distance_dict[current].values()))[0]
                walk_step = walk_step+1
                trajectory.append(copy.deepcopy(current))
            corpus.append(copy.deepcopy(trajectory))
            d['id'].append(id1)
            d['num'].append(i)
            id2 = trajectory[-1]
            lat1 = placeid_df.loc[id1]['lat']
            lon1 = placeid_df.loc[id1]['lon']
            lat2 = placeid_df.loc[id2]['lat']
            lon2 = placeid_df.loc[id2]['lon']
            d['distance'].append(getdistance(lon1, lat1, lon2, lat2))
    print('random walk complete')
    d = pd.DataFrame(d)
    print('random walk info:')
    print(d['distance'].describe())

    print('spatial pretraining start...')
    t0 = time.time()
    model = gensim.models.Word2Vec(corpus, min_count=1, workers=10, window=window, vector_size=embedding_dim, epochs=epochs, sg=sg)
    model.save(save_path)
    t1 = time.time()
    print(f'pretraining spatial complete, {t1-t0} seconds, dict size={len(model.wv.key_to_index)}, corpus size={len(corpus)}')
    return model


def distance2weight(d):
    return math.exp(-d*d)
