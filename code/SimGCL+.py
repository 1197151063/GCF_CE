from model import RecModel
from torch import LongTensor,Tensor
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
from dataloader import Loader
import world
from procedure import test
import utils
import torch.nn.functional as F
path='../data/' + world.config['dataset']
if world.config['dataset'] == 'yelp2018':
    config = {
        'init_weight':2,
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'ssl_tmp':0.2,#TEMPERATURE
        'ssl_decay':0.5,#SSL_STRENGTH
        'aug_ratio':0.1,#NOISE_RATIO
        'alpha':1e-1,
    }
if world.config['dataset'] == 'amazon-book':
    config = {
        'init_weight':2,
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'ssl_tmp':0.2,#TEMPERATURE
        'ssl_decay':2,#SSL_STRENGTH
        'aug_ratio':0.1,#NOISE_RATIO
        'alpha':1e-1,
    }
if world.config['dataset'] == 'gowalla':
    config = {
        'init_weight':2,
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'ssl_tmp':0.2,#TEMPERATURE
        'ssl_decay':0.05,#SSL_STRENGTH
        'aug_ratio':0.2,#NOISE_RATIO
        'alpha':1e-1,
    }


class SimGCL(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 knn_ind,
                 value,
                 config):
        super().__init__(num_users=num_users,
                         num_items=num_items,
                         edge_index=edge_index,
                         config=config
                         )
        self.K = config['K']
        self.num_interactions = edge_index.size(1)
        self.train_edge_index = edge_index
        self.edge_index = self.get_sparse_graph(edge_index,use_value=False,value=None)
        self.edge_index = gcn_norm(self.edge_index)        
        self.alpha= 1./ (self.K)
        self.knn_ind = knn_ind
        self.val = value
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['dim'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['dim'])
        #SimGCL use Uniform initilizer
        nn.init.xavier_uniform_(self.user_emb.weight,gain=config['init_weight'])
        nn.init.xavier_uniform_(self.item_emb.weight,gain=config['init_weight'])
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.K
        else:
            self.alpha = torch.tensor([self.alpha] * (self.K))
        self.ssl_tmp = config['ssl_tmp']
        self.ssl_decay = config['ssl_decay']
        self.eps = config['aug_ratio']
        print('Go backbone SimGCL')
        print(f"params settings: \n L2 reg:{config['decay']}\n layer:{self.K}")
        print(f" ssl_tmp:{config['ssl_tmp']}\n ssl_decay:{config['ssl_decay']}\n noise_bias:{config['aug_ratio']}")
    
    def norm(self,x):
        users,items = torch.split(x,[self.num_users,self.num_items])
        users_norm = (1e-6 + users.pow(2).sum(dim=1).mean()).sqrt()
        items_norm = (1e-6 + items.pow(2).sum(dim=1).mean()).sqrt()
        users = users / (items_norm)
        items = items / (users_norm)
        x = torch.cat([users,items])
        return x
    
    def get_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = []
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            out.append(x)
        out = torch.stack(out,dim=1)
        out = torch.mean(out,dim=1)
        return out
    
    def get_shuffle_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = []
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            r_noise = torch.rand_like(x).cuda()
            x = x + torch.sign(x) * F.normalize(r_noise,dim=-1) * self.eps
            out.append(x)
        out = torch.stack(out,dim=1)
        out = torch.mean(out,dim=1)
        return out
    
    def ssl_loss(self,edge_label_index):
        u_idx,i_idx = edge_label_index[0],edge_label_index[1]
        view1 = self.get_shuffle_embedding()
        view2 = self.get_shuffle_embedding()
        info_out_u_1,info_out_i_1 = torch.split(view1,[self.num_users,self.num_items])
        info_out_u_2,info_out_i_2 = torch.split(view2,[self.num_users,self.num_items])
        u_idx = torch.unique(edge_label_index[0])
        i_idx = torch.unique(edge_label_index[1])
        user_cl_loss = utils.InfoNCE(info_out_u_1[u_idx], info_out_u_2[u_idx], 0.2)
        item_cl_loss = utils.InfoNCE(info_out_i_1[i_idx], info_out_i_2[i_idx], 0.2)
        return self.ssl_decay * (user_cl_loss + item_cl_loss)    

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
    
    def bpr_loss(self,pos_rank,neg_rank):
        return F.softplus(neg_rank - pos_rank).mean()
    
    def L2_reg(self,edge_label_index):
        u_idx,i_idx_pos,i_idx_neg = edge_label_index
        userEmb0 = self.user_emb.weight[u_idx]
        posEmb0 = self.item_emb.weight[i_idx_pos]
        negEmb0 = self.item_emb.weight[i_idx_neg]
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2)) / edge_label_index.size(1)
        regularization = self.config['decay'] * reg_loss
        return regularization

def train(dataset,model:SimGCL,opt):
    model = model
    model.train()
    S = utils.Fast_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(S)
    for edge_label_index in S:
        pos_rank,neg_rank = model(edge_label_index)
        bpr_loss = model.bpr_loss(pos_rank,neg_rank)
        ssl_loss = model.ssl_loss(edge_label_index)
        i_i_loss = model.item_constraint_loss(edge_label_index)
        L2_reg = model.L2_reg(edge_label_index)
        loss = bpr_loss + ssl_loss + L2_reg + i_i_loss
        opt.zero_grad()
        loss.backward()
        opt.step()    
        aver_loss += (bpr_loss + ssl_loss + L2_reg + i_i_loss)
    aver_loss /= total_batch
    return f"average loss {aver_loss:5f}"
device = world.device
dataset = Loader()
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
value = torch.load(path+'/i-i-value.pt')
knn_index = torch.load(path+'/i-i-index.pt')
value = value[:,1:].to(device)
value[value < 0.1] = 0
knn_index = knn_index[:,1:].to(device)
model = SimGCL(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 knn_ind=knn_index,
                 value=value,
                 config=config).to(device)
opt = torch.optim.AdamW(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.

for epoch in range(1, 1001):
    loss = train(dataset=dataset,
                         model=model,
                         opt=opt)
    recall,ndcg = test([20,50],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if flag == 1:
        break
    print(f'Epoch: {epoch:03d}, {loss}, R@20: '
          f'{recall[20]:.4f}, R@50: {recall[50]:.4f} '
          f', N@20: {ndcg[20]:.4f}, N@50: {ndcg[50]:.4f}')