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
        'gamma':1,#UNIFORMITY
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
        'gamma':5,#UNIFORMITY
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
        'gamma':0.8,#UNIFORMITY
        'alpha':1e-1,
    }
class DirectAU(RecModel):
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
        self.K = config['K']
        edge_index = self.get_sparse_graph(edge_index=edge_index,use_value=False,value=None)
        self.edge_index = gcn_norm(edge_index)
        self.alpha= 1./ (1 + self.K)
        self.gamma = config['gamma']
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.K + 1
        else:
            self.alpha = torch.tensor([self.alpha] * (self.K + 1))
        print('Go DirectAU')
        print(f"params settings: \n emb_size:{config['dim']}\n L2 reg:{config['decay']}\n layer:{self.K}\n uniformity:{config['gamma']}")

    def init_weight(self,init_weight):
        nn.init.xavier_uniform_(self.user_emb.weight,gain=init_weight)
        nn.init.xavier_uniform_(self.item_emb.weight,gain=init_weight)
        
    def get_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            out = out + x * self.alpha[i + 1]
        return out
    
    def alignment(self,x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(dim=1).pow(2).mean()

    def alignment_loss(self,edge_label_index):
        out = self.get_embedding()
        x_u,x_i = torch.split(out,[self.num_users,self.num_items])
        batch_x_u,batch_x_i = x_u[edge_label_index[0]],x_i[edge_label_index[1]]
        return self.alignment(batch_x_u,batch_x_i)
    
    def uniformity_loss(self,edge_label_index):
        out = self.get_embedding()
        x_u,x_i = torch.split(out,[self.num_users,self.num_items])
        batch_x_u,batch_x_i = x_u[edge_label_index[0]],x_i[edge_label_index[1]]
        return  self.gamma * (self.uniformity(batch_x_u) + self.uniformity(batch_x_i))

    def uniformity(self,x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    
    def L2_reg(self,edge_label_index):
        u_idx,i_idx_pos,i_idx_neg = edge_label_index
        userEmb0 = self.user_emb.weight[u_idx]
        posEmb0 = self.item_emb.weight[i_idx_pos]
        negEmb0 = self.item_emb.weight[i_idx_neg]
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2)) / edge_label_index.size(1)
        regularization = self.config['decay'] * reg_loss
        return regularization
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


def train_dau(dataset,
                  model:DirectAU,
                  opt):
    model = model
    model.train()
    S = utils.Fast_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(S)
    for edge_label_index in S:
        align = model.alignment_loss(edge_label_index)
        uniform = model.uniformity_loss(edge_label_index)
        L2_reg = model.L2_reg(edge_label_index)
        i_i_loss = model.item_constraint_loss(edge_label_index)
        loss = align + uniform + L2_reg + i_i_loss
        opt.zero_grad()
        loss.backward()
        opt.step()    
        aver_loss += (align + uniform + i_i_loss + L2_reg)
    aver_loss /= total_batch
    return f"average loss {aver_loss:5f}"


device = world.device
dataset = Loader()
utils.set_seed(config['seed'])
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
value = torch.load(path+'/i-i-value.pt')
knn_index = torch.load(path+'/i-i-index.pt')
value = value[:,1:].to(device)
value[value < 0.1] = 0
knn_index = knn_index[:,1:].to(device)
model = DirectAU(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config).to(device)
opt = torch.optim.AdamW(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.
for epoch in range(1, 1001):
    loss = train_dau(dataset=dataset,model=model,opt=opt)
    recall,ndcg = test([20,50],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if flag == 1:
        break
    print(f'Epoch: {epoch:03d}, {loss}, R@20: '
          f'{recall[20]:.4f}, R@50: {recall[50]:.4f} '
          f', N@20: {ndcg[20]:.4f}, N@50: {ndcg[50]:.4f}')