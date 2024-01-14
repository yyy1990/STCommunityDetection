import math
import joblib
import numpy as np
from tqdm import tqdm
from transbigdata import getdistance
from sklearn.neighbors import KDTree


def distance2weight(d):
    return 9*math.exp(-d*d)+1


def topn_spatial_graph(check_in, topn, d_thre):
    placeid_df = check_in.groupby('placeid').head(1)
    placeid_df = placeid_df[['placeid', 'lat', 'lon']]

    placeid_list = list(placeid_df['placeid'].unique())
    placeid_df.set_index(keys='placeid', inplace=True)

    print('getting coordinates...') #time: 1min
    coord = []
    for id1 in tqdm(placeid_list):
        lat = placeid_df.loc[id1]['lat']
        lon = placeid_df.loc[id1]['lon']
        coord.append([lat, lon])
    coord = np.array(coord)
    tree = KDTree(coord, leaf_size=topn)
    print('KD tree constructed')

    ans = {}
    print('getting spatial edges...')
    for p1 in tqdm(placeid_list): #time:
        lat1 = placeid_df.loc[p1]['lat']
        lon1 = placeid_df.loc[p1]['lon']
        _, ind = tree.query(np.array([lat1, lon1]).reshape(1, -1), k=topn+1)
        for i in range(len(ind[0])):
            p2 = placeid_list[ind[0][i]]
            if p2!=p1:
                lat2 = placeid_df.loc[p2]['lat']
                lon2 = placeid_df.loc[p2]['lon']
                d = getdistance(lon1, lat1, lon2, lat2)
                if d<=d_thre:
                    if p1 not in ans:
                        ans[p1] = {}
                    ans[p1][p2] = distance2weight(d)
    return ans


def generate_global_graph(check_in, t_thre, d_thre, topn, node_save='gowalla_node_list.pkl', temporal_save='gowalla_temporal_edge.pkl', spatial_save='gowalla_spatial_edge.pkl'):
    userid_list = list(check_in['userid'].unique())
    check_in_group = check_in.groupby('userid')
    node_list = list(check_in['placeid'].unique())
    joblib.dump(node_list, node_save)
    print('node list saved')

    spatial_edge = topn_spatial_graph(check_in, topn, d_thre)
    joblib.dump(spatial_edge, spatial_save)
    print('spatial saved')

    temporal_edge = {}
    print('getting temporal edges...')
    for user in tqdm(userid_list):
        user_data = check_in_group.get_group(user)
        for i, row in enumerate(user_data.iterrows()):
            if i+1<len(user_data):
                t1 = user_data.iloc[i]['timestamp']
                p1 = user_data.iloc[i]['placeid']
                t2 = user_data.iloc[i+1]['timestamp']
                p2 = user_data.iloc[i+1]['placeid']
                t_delta = abs(t2-t1)
                if t_delta<=t_thre:
                    if p1 not in temporal_edge:
                        temporal_edge[p1] = {}

                    if p2 not in temporal_edge[p1]:
                        temporal_edge[p1][p2] = 1
                    else:
                        temporal_edge[p1][p2] += 1
    joblib.dump(temporal_edge, temporal_save)
    print('temporal saved')

    return node_list, spatial_edge, temporal_edge
