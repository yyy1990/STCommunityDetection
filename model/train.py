import time
import torch
import gensim
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset
from augmented_trajectory import random_aug_walk
from net import GGT_net
from torch.utils.tensorboard import SummaryWriter


class index_dataset(Dataset):
    def __init__(self, check_in, train=True):
        self.train = train
        self.user_id = list(check_in['userid'].unique())

    def __len__(self):
        if self.train:
            return len(self.user_id)
        else:
            return 1

    def __getitem__(self, index):
        if self.train:
            return self.user_id[index]
        else:
            return self.user_id
        

class graph_dataset(Dataset):
    def __init__(self, check_in, placeid_model, spatial_model, spatial_edge, temporal_edge, train=False, select_label=0.2, spatial_add_prob=0.5,
    add_prob=0.5, min_len=5, max_len=4096, aug_num=20, topn=3, embedding_selector='both', user_embedding_init='both'):
        self.check_in_group = check_in.groupby('userid')
        self.placeid_model = placeid_model
        self.spatial_model = spatial_model
        self.train = train
        self.select_label = select_label
        self.topn = topn

        #parameter for data augumentation
        self.spatial_edge = spatial_edge
        self.temporal_edge = temporal_edge
        self.spatial_add_prob = spatial_add_prob
        self.add_prob = add_prob
        self.min_len = min_len
        self.max_len = max_len
        self.aug_num = aug_num

        self.embedding_selector = embedding_selector
        self.user_embedding_init = user_embedding_init

    def __len__(self):
        if not self.train:
            return 1
        else:
            return self.check_in_group.ngroups

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_aug_sample(self, seed_p):
        seed_p = random_aug_walk(seed_p, self.spatial_edge, self.temporal_edge, self.spatial_add_prob, self.add_prob, self.min_len, self.max_len, self.aug_num)
        return seed_p

    def get_sample_data(self, spatial_seq, user, topn=3):
        placeid_seq = [str(i) for i in spatial_seq]
        placeid_counter = Counter(placeid_seq)
        top_n = placeid_counter.most_common(topn)
        placeid_embedding = self.placeid_model.wv[placeid_seq]
        spatial_embedding = self.spatial_model.wv[spatial_seq]
        if self.embedding_selector == 'both':
            arr = np.hstack([placeid_embedding, spatial_embedding])
        elif self.embedding_selector == 'semantic':
            arr = placeid_embedding
        elif self.embedding_selector == 'spatial':
            arr = spatial_embedding

        topn_place = []
        for p, _ in top_n:
            topn_place.append(p)
        placeid_seq_topn = [i for i in placeid_seq if i in topn_place]
        spatial_seq_topn = placeid_seq_topn

        placeid_embedding_topn = self.placeid_model.wv[placeid_seq_topn]
        spatial_embedding_topn = self.spatial_model.wv[spatial_seq_topn]
        if self.embedding_selector == 'both':
            arr_topn = np.hstack([placeid_embedding_topn, spatial_embedding_topn])
        elif self.embedding_selector == 'semantic':
            arr_topn = placeid_embedding_topn
        elif self.embedding_selector == 'spatial':
            arr_topn = spatial_embedding_topn

        if len(placeid_seq)>0:
            arr = np.mean(arr, axis=0)
            arr_topn = np.mean(arr_topn, axis=0)

        if self.user_embedding_init == 'both':
            arr = (arr+arr_topn)/2
        elif self.user_embedding_init == 'global':
            arr = arr
        elif self.user_embedding_init == 'local':
            arr = arr_topn 

        edges = [{'user':user, 'place':e, 'number': placeid_counter[e]} for e in placeid_counter]
        return arr, placeid_seq, edges

    def get_sample(self, index):
        user_list = index
        placeid_list = []
        sptial_list = []
        node_embedding = []
        edge_df = []
        real_user_list = []

        for i in user_list:
            data = self.check_in_group.get_group(i)
            spatial_seq = list(data['placeid'])
            arr, placeid_seq, edges = self.get_sample_data(spatial_seq, i)
            node_embedding.append(arr)
            placeid_list = placeid_list+placeid_seq
            sptial_list = sptial_list+spatial_seq
            real_user_list.append(i)
            edge_df = edge_df+edges

            if self.train:
                seed_p = random_aug_walk(spatial_seq, self.spatial_edge, self.temporal_edge, self.spatial_add_prob, self.add_prob, self.min_len, self.max_len,
                self.aug_num)
                user = str(i)+'-'+'aug'
                arr, placeid_seq, edges = self.get_sample_data(seed_p, user)
                node_embedding.append(arr)
                placeid_list = placeid_list+placeid_seq
                sptial_list = sptial_list+seed_p
                real_user_list.append(user)
                edge_df = edge_df+edges

        number_of_user = len(real_user_list)
        placeid_list = list(set(placeid_list))
        sptial_list = list(set(sptial_list))
        user_dict = {user:ind for ind, user in enumerate(real_user_list)}
        placeid_dict = {placeid:ind+number_of_user for ind,placeid in enumerate(placeid_list)}
        number_of_place = len(placeid_list)
        node_embedding = np.vstack(node_embedding)

        place_embedding = self.placeid_model.wv[placeid_list]
        spatial_embedding = self.spatial_model.wv[sptial_list]
        if self.embedding_selector == 'both':
            placeid_embeddding = np.hstack([place_embedding, spatial_embedding])
        elif self.embedding_selector == 'semantic':
            placeid_embeddding = place_embedding
        elif self.embedding_selector == 'spatial':
            placeid_embeddding = spatial_embedding
        node_embedding = np.vstack([node_embedding, placeid_embeddding])

        edge_df = pd.DataFrame(edge_df)
        edge_df['user'] = edge_df['user'].map(lambda x: user_dict[x])
        edge_df['place'] = edge_df['place'].map(lambda x: placeid_dict[x])

        if self.train:
            select_label = int(self.select_label*len(edge_df))

            edge_df_target = edge_df.sample(n=select_label, replace=False)
            target_user = set(list(edge_df_target['user'].unique()))
            target_place = set(list(edge_df_target['place'].unique()))
            edge_df['select'] = edge_df.apply(lambda x: True if x['user'] in target_user and x['place'] in target_place else False, axis=1)
            edge_df_target = edge_df[edge_df['select']==True]
            edge_df = edge_df[~edge_df.index.isin(edge_df_target.index)]

            edge_index = np.array(edge_df[['user', 'place']]).T
            edge_attr = np.array(edge_df['number'], dtype=np.float32)

            target_user = list(edge_df_target['user'])
            target_place = list(edge_df_target['place'])

            return node_embedding, edge_index, edge_attr, target_user, target_place

        else:
            edge_index = np.array(edge_df[['user', 'place']]).T
            edge_attr = np.array(edge_df['number'], dtype=np.float32)
            return node_embedding, edge_index, edge_attr


def train(check_in,batch_size,place_id_embedding_model,spatial_embedding_model,spatial_edge,temporal_edge, input_dim, output_dim, heads, dropout, 
          device, lr, label_smoothing, save_prefix, epoch_num, check_point_interval):
    check_in = pd.read_csv(check_in)
    place_id_embedding_model = gensim.models.Word2Vec.load(place_id_embedding_model)
    spatial_embedding_model = gensim.models.Word2Vec.load(spatial_embedding_model)
    spatial_edge = joblib.load(spatial_edge)
    temporal_edge = joblib.load(temporal_edge)

    idx_set = index_dataset(check_in, train=True)
    train_dataloader = torch.utils.data.DataLoader(dataset=idx_set, batch_size=batch_size, shuffle=True, num_workers=8)
    g_set = graph_dataset(check_in, place_id_embedding_model, spatial_embedding_model, spatial_edge, temporal_edge, train=True, select_label=0.2, aug_num=10,
    spatial_add_prob=0.5, topn=3)

    if torch.cuda.is_available():
        use_gpu = True
        print('training on GPU mode')
    else:
        use_gpu = False
        print('training on CPU mode')

    net = GGT_net(input_dim, output_dim, heads, dropout, device)
    print('model structure:')
    print(net)

    opt = torch.optim.Adam(net.parameters(), lr)
    cost = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if use_gpu:
        net = net.to(device=device)
        cost = cost.to(device=device)

    writer = SummaryWriter(log_dir=save_prefix)

    global_step = 0
    print_n = int(len(train_dataloader)/10)
    t_sum = 0

    for epoch in range(epoch_num):
        t0 = time.time()
        running_loss = 0.0
        print("-"*10)
        print(f'Epoch {epoch+1}/{epoch_num}')

        for data in tqdm(train_dataloader):
            node_embedding, edge_index, edge_attr, target_user, target_place = g_set[data.tolist()]

            node_embedding = torch.tensor(node_embedding)
            edge_index = torch.tensor(edge_index)
            edge_attr = torch.tensor(edge_attr)
            label = torch.tensor([i for i in range(len(target_user))])
            user_target = []
            for i in range((len(data)*2)):
                if i%2==0:
                    user_target.append(i+1)
                else:
                    user_target.append(i-1)
            user_target = torch.tensor(user_target)

            if use_gpu:
                node_embedding = node_embedding.to(device=device)
                edge_index = edge_index.to(device=device)
                edge_attr = edge_attr.to(device=device)
                label = label.to(device=device)
                user_target = user_target.to(device=device)

            x = net(node_embedding, edge_index, edge_attr)

            target_user_output = x[target_user]
            target_place_output = x[target_place]

            user_output = x[0:(len(data)*2)]
            sim_user = torch.matmul(user_output, user_output.T)
            sim_user_diag = torch.diag(sim_user)
            sim_user_diag = torch.diag_embed(sim_user_diag)
            sim_user = sim_user-sim_user_diag

            sim_edge = torch.matmul(target_user_output, target_place_output.T)
            loss_mask = cost(sim_edge, label)
            loss_user_aug = cost(sim_user, user_target)
            loss = loss_mask+loss_user_aug

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.data.item()

            writer.add_scalar('BatchLoss/edgeMaskLoss', loss_mask.data.item(), global_step=global_step)
            writer.add_scalar('BatchLoss/userSimLoss', loss_user_aug.data.item(), global_step=global_step)
            writer.add_scalar('BatchLoss/LossSum', loss.data.item(), global_step=global_step)
            global_step = global_step+1

        if epoch%check_point_interval == 0:
            torch.save(net.state_dict(), f'{save_prefix}net_parameter-epoch:{epoch}.pth')
        epoch_loss = running_loss/len(train_dataloader)
        writer.add_scalar('EpochLoss', epoch_loss, global_step=epoch)
        t1 = time.time()
        print(f'Loss is {epoch_loss}, takes {t1-t0} seconds')
        t_sum = t_sum+t1-t0
    torch.save(net.state_dict(), f'{save_prefix}net_parameter.pth')
    print(f'training takes {t_sum} seconds')