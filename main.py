from preprocessing import checkin, pretrain, graph
from config import gowalla, brightkite, weeplaces


def pipeline(c):
    #preprocess check-in
    check_in_preprocess = checkin.preprocess_totalcheckin(c.origin_check_in, c.check_in_remove_sparseplace, c)
    check_in_preprocess = checkin.checkin_preprocess(c.check_in_remove_sparseplace, c.check_in_preprocess).df

    #pretrain
    pretrain.pretrain_embedding(check_in_preprocess, sort_col='timestamp', id_col='userid', target_col='placeid', embedding_dim=c.embedding_dim_semantic,
                                save_path=c.pretrain_semantic_path, sg=1, min_count=1, workers=10, window=5, epochs=10, min_split=10, split=True,
                                real_sequence=True)

    pretrain.pretrain_spatial_embedding(walk_num=c.spatial_walk_num, walk_len=c.spatial_walk_len, potential_nerghbor=c.spatial_walk_potential_nerghbor,
                                        embedding_dim=c.embedding_dim_spatial, save_path=c.pretrain_spatial_path, check_in_path=c.check_in_preprocess, sg=0)

    graph.generate_global_graph(check_in=c.check_in_preprocess, t_thre=c.t_thre, d_thre=c.d_thre, topn=c.topn, node_save=c.node_path, 
                                temporal_save=c.temporal_path, spatial_save=c.spatial_path)



if __name__=='__main__':
    pipeline(gowalla)
    pipeline(brightkite)
    pipeline(weeplaces)
