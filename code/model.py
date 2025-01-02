from torch import nn,Tensor,LongTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import SparseTensor
from torch_sparse import SparseTensor,matmul
import torch
import world
device = world.device
"""
Already supported:[LightGCN, SGL, SimGCL, NCL, DirectAU]
"""

class RecModel(MessagePassing):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 config,
                 edge_index:LongTensor):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.config = config
        self.f = nn.Sigmoid()

    def get_sparse_heter(self,edge_index,val):
        num_nodes = edge_index[0].max()+1
        return SparseTensor(row=edge_index[0],col=edge_index[1],
                            value=val,sparse_sizes=(num_nodes,num_nodes))

    def get_sparse_graph(self,
                         edge_index,
                         use_value=False,
                         value=None):
        num_users = self.num_users
        num_nodes = self.num_nodes
        r,c = edge_index
        row = torch.cat([r , c + num_users])
        col = torch.cat([c + num_users , r])
        if use_value:
            value = torch.cat([value,value])
            return SparseTensor(row=row,col=col,value=value,sparse_sizes=(num_nodes,num_nodes))
        else:
            return SparseTensor(row=row,col=col,sparse_sizes=(num_nodes,num_nodes))
    
    def get_embedding(self):
        raise NotImplementedError
    
    def forward(self,
                edge_label_index:Tensor):
        out = self.get_embedding()
        out_u,out_i = torch.split(out,[self.num_users,self.num_items])
        out_src = out_u[edge_label_index[0]]
        out_dst = out_i[edge_label_index[1]]
        out_dst_neg = out_i[edge_label_index[2]]
        return (out_src * out_dst).sum(dim=-1),(out_src * out_dst_neg).sum(dim=-1)
    
    def link_prediction(self,
                        src_index:Tensor=None,
                        dst_index:Tensor=None):
        out = self.get_embedding()
        out_u,out_i = torch.split(out,[self.num_users,self.num_items])
        if src_index is None:
            src_index = torch.arange(self.num_users).long()
        if dst_index is None:
            dst_index = torch.arange(self.num_items).long()
        out_src = out_u[src_index]
        out_dst = out_i[dst_index]
        pred = out_src @ out_dst.t()
        return pred
    
    # BPR+L2
    def recommendation_loss(self,
                            pos_rank,
                            neg_rank,
                            edge_label_index):
        rec_loss = torch.nn.functional.softplus(neg_rank - pos_rank).mean()
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        embedding = torch.cat([user_emb[edge_label_index[0]],
                               item_emb[edge_label_index[1]],
                               item_emb[edge_label_index[2]]])
        regularization = self.config['decay'] * (1/2) * embedding.norm(p=2).pow(2)
        regularization = regularization / pos_rank.size(0)
        return rec_loss , regularization
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t,x)
    


