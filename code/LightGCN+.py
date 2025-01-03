from model import RecModel
from torch import LongTensor,Tensor
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
from dataloader import Loader
import world
from procedure import test
import utils
device = world.device
path='../data/' + world.config['dataset']
if world.config['dataset'] == 'yelp2018':
    config = {
        'init_weight':1e-4,
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'alpha':1e-1,#CONSTRAINT_LOSS
    }

if world.config['dataset'] == 'amazon-book':
    config = {
        'init_weight':1e-2,
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
    }

if world.config['dataset'] == 'gowalla':
    config = {
        'init_weight':1e-2,
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
    }

class LightGCN(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 knn_ind,
                 value,
                 config):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            config=config,
            edge_index=edge_index
        )
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['dim'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['dim'])
        self.init_weight(config['init_weight'])
        self.knn_ind = knn_ind
        self.val = value
        self.K = config['K']
        # self.sp_edge_index = self.graph_sparsify(edge_index=edge_index)
        # self.sp_edge_index = self.get_sparse_graph(self.sp_edge_index)
        # print(self.sp_edge_index)
        # self.sp_edge_index = gcn_norm(self.sp_edge_index)
        edge_index = self.get_sparse_graph(edge_index=edge_index,use_value=False,value=None)
        self.edge_index = gcn_norm(edge_index)
        self.alpha= 1./ (1 + self.K)
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.K + 1
        else:
            self.alpha = torch.tensor([self.alpha] * (self.K + 1))
        print('Go LightGCN')
        print(f"params settings: \n emb_size:{config['dim']}\n L2 reg:{config['decay']}\n layer:{self.K}")

    def init_weight(self,init_weight):
        nn.init.normal_(self.user_emb.weight,std=init_weight)
        nn.init.normal_(self.item_emb.weight,std=init_weight)
    # def graph_sparsify(self,edge_index):
    #     user_d = degree(edge_index[0], num_nodes=self.num_users)
    #     num_h_users = int(self.num_users * 0.2)
    #     h_users = torch.topk(user_d, num_h_users).indices
    #     mask = torch.isin(edge_index[0], h_users)
    #     head_user_edges = edge_index[:, mask]
    #     other_edges = edge_index[:, ~mask]
    #     num_edges_to_drop = int(0.1 * edge_index.size(1))
    #     if num_edges_to_drop > 0:
    #         perm = torch.randperm(head_user_edges.size(1))
    #         drop_indices = perm[:num_edges_to_drop]
    #         keep_mask = torch.ones(head_user_edges.size(1), dtype=torch.bool, device=head_user_edges.device)
    #         keep_mask[drop_indices] = False
    #         head_user_edges = head_user_edges[:, keep_mask]
    #     new_edge_index = torch.cat((other_edges, head_user_edges), dim=1)
            
    #     return new_edge_index
    
    def get_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            out = out + x * self.alpha[i + 1]
        return out
    
    def item_alignment(self,items):
        item_emb = self.item_emb.weight
        knn_neighbour = self.knn_ind[items] #[batch_size * k]
        # sim_score = self.val[items]
        user_emb = item_emb[items].unsqueeze(1)
        item_emb_pos = item_emb[knn_neighbour] 
        loss =  (user_emb * item_emb_pos).sum(dim=-1)
        return -loss.sigmoid().log().sum()
    
    def item_constraint_loss(self,edge_label_index):
        pos = edge_label_index[1]
        i_loss = self.item_alignment(pos)
        return  config['alpha'] * i_loss
    
    def get_embedding_with_edge_index(self,edge_index):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=edge_index,x=x)
            out = out + x * self.alpha[i + 1]
        return out

    def hn_loss(self,edge_label_index):
        users,items,_ = edge_label_index
        emb1 = self.get_embedding()
        emb2 = self.get_embedding_with_edge_index(self.sp_edge_index)
        # # emb2 = self.linear(emb2)
        x1,y1 = torch.split(emb1,[self.num_users,self.num_items])
        x2,y2 = torch.split(emb2,[self.num_users,self.num_items])
        # x1,y1 = x1[users],y1[items]
        # x2,y2 = x2[users],y2[items]
        # x1, y1 = F.normalize(x1, dim=-1), F.normalize(y1, dim=-1)
        # x2, y2 = F.normalize(x2, dim=-1), F.normalize(y2,dim=-1)
        # return ((x1 - y2).norm(dim=1).pow(2).mean() + (x2 - y1).norm(dim=1).pow(2).mean()) * 1/2
        a1 = self.InfoNCE_U_ALL(x1,y2,users,items,0.1)
        a2 = self.InfoNCE_U_ALL(x2,y1,users,items,0.1)
        return 0.1 * (a1 + a2)
    
    
    def forward(self,
                    edge_label_index:Tensor):
            out = self.get_embedding()
            out_u,out_i = torch.split(out,[self.num_users,self.num_items])
           
            out_src = out_u[edge_label_index[0]]
            out_dst = out_i[edge_label_index[1]]
            out_dst_neg = out_i[edge_label_index[2]]
            return (out_src * out_dst).sum(dim=-1),(out_src * out_dst_neg).sum(dim=-1)
    


def train(dataset,model:LightGCN,opt):
    model = model
    model.train()
    S = utils.Fast_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(S)
    for edge_label_index in S:
        opt.zero_grad()
        pos_rank,neg_rank = model(edge_label_index)
        bpr_loss,reg_loss = model.recommendation_loss(pos_rank,neg_rank,edge_label_index)
        i_i_loss = model.item_constraint_loss(edge_label_index)
        loss = bpr_loss + reg_loss + i_i_loss
        loss.backward()
        opt.step()   
        aver_loss += (loss)
    aver_loss /= total_batch
    return f"average loss {aver_loss:5f}"
    
value = torch.load(path+'/i-i-value.pt')
knn_index = torch.load(path+'/i-i-index.pt')
value = value[:,1:].to(device)
value[value < 0.1] = 0
knn_index = knn_index[:,1:].to(device)
dataset = Loader()
utils.set_seed(config['seed'])
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model = LightGCN(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config,
                 knn_ind=knn_index,
                 value=value).to(device)
opt = torch.optim.AdamW(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.
for epoch in range(1, 1001):
    loss = train(dataset=dataset,model=model,opt=opt)
    recall,ndcg = test([20,50],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if flag == 1:
        break
    print(f'Epoch: {epoch:03d}, {loss}, R@20: '
          f'{recall[20]:.4f}, R@50: {recall[50]:.4f} '
          f', N@20: {ndcg[20]:.4f}, N@50: {ndcg[50]:.4f}')