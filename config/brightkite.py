import os
result_path = './results/brightkite/'
folder = os.path.exists(result_path)
if not folder:
        os.makedirs(result_path)

#preprocessing config
origin_check_in = './dataset/brightkite/Brightkite_totalCheckins.txt'
check_in_remove_sparseplace = result_path+'Brightkite_remove_sparse.csv'
sparse_place_thre = 3
check_in_preprocess = result_path+'Brightkite_preprocessed_check_in.csv'
