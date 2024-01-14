from preprocessing import checkin, pretrain, graph
from config import gowalla, brightkite, weeplaces
from model import train
from model import evaluate


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

    #train and evaluate
    train.train(check_in=c.check_in_preprocess,batch_size=c.batch_size,place_id_embedding_model=c.pretrain_semantic_path,spatial_embedding_model=c.pretrain_spatial_path,
                spatial_edge=c.spatial_path,temporal_edge=c.temporal_path, input_dim=c.input_dim, output_dim=c.output_dim_head, heads=c.heads, dropout=c.dropout, 
                device=c.device, lr=c.lr, label_smoothing=c.label_smoothing, save_prefix=c.result_path, epoch_num=c.epoch_num, check_point_interval=c.check_point_interval)

    evaluate.evaluate(check_in=c.check_in_preprocess, label_edge=c.label_edge, save_prefix=c.result_path, evaluate_report_path=c.evaluate_report_path, batch_size=c.batch_size,
                      place_id_embedding_model=c.pretrain_semantic_path, spatial_embedding_model=c.pretrain_spatial_path, spatial_edge=c.spatial_path, temporal_edge=c.temporal_path,
                      input_dim=c.input_dim, output_dim=c.output_dim_head, headss=c.heads, device=c.device, select_label=0.2, topn_select=3)


if __name__=='__main__':
    pipeline(gowalla)
    pipeline(brightkite)
    pipeline(weeplaces)
