import world
from dataloader import Loader
import torch
from tqdm import tqdm
"""
COMMON NEIGHBOUR RATIO
(JACCARD SIM)
"""
device = world.device
dataset = Loader()
num_users = dataset.num_users           
num_items = dataset.num_items
t_edge_index = dataset.train_edge_index
adj_mat = torch.sparse_coo_tensor(t_edge_index,torch.ones(t_edge_index[0].size(0),dtype=torch.int8),(num_users,num_items)).to_dense().T
# # Step 3: 分块计算共现矩阵
# batch_size = 1024
# num_batches = (num_items + batch_size - 1) // batch_size
# item_item_common = torch.zeros((num_items, num_items))

# for i in tqdm(range(num_batches)):
#     start = i * batch_size
#     end = min((i + 1) * batch_size, num_items)
#     item_item_common[start:end] = adj_mat[start:end] @ adj_mat.T
item_item_common = adj_mat @ adj_mat.T
item_user_counts = adj_mat.sum(dim=1)
all_neighbors_matrix = item_user_counts.view(-1, 1) + item_user_counts.view(1, -1) - item_item_common
i_i_co = item_item_common / all_neighbors_matrix.float()
value, knn_ind = torch.topk(i_i_co, 11, dim=-1)
torch.save(value,'i-i-value.pt')
torch.save(knn_ind,'i-i-index.pt')

