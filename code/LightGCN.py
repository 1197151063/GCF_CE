from model import RecModel
from torch import LongTensor,Tensor
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
from dataloader import Loader
import world
from procedure import train_bpr,test
import utils

config = {
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
        self.init_weight(1e-1)
        self.K = config['K']
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
        
    def get_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            out = out + x * self.alpha[i + 1]
        return out
    
    def forward(self,
                    edge_label_index:Tensor):
            out = self.get_embedding()
            out_u,out_i = torch.split(out,[self.num_users,self.num_items])
            out_src = out_u[edge_label_index[0]]
            out_dst = out_i[edge_label_index[1]]
            out_dst_neg = out_i[edge_label_index[2]]
            return (out_src * out_dst).sum(dim=-1),(out_src * out_dst_neg).sum(dim=-1)
    

device = world.device
dataset = Loader()
utils.set_seed(config['seed'])
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model = LightGCN(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.
for epoch in range(1, 1001):
    loss = train_bpr(dataset=dataset,model=model,opt=opt)
    recall,ndcg = test([20,50],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if flag == 1:
        break
    print(f'Epoch: {epoch:03d}, {loss:.4f}, R@20: '
          f'{recall[20]:.4f}, R@50: {recall[50]:.4f} '
          f', N@20: {ndcg[20]:.4f}, N@50: {ndcg[50]:.4f}')