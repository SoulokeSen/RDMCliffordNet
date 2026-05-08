import torch
import torch
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool
import torch.nn as nn
from algebra.cliffordalgebra import CliffordAlgebra
from cegnn_utils import MVLinear, MVLayerNorm

import torch
from torch import nn
import utils
#from losses import RDM2sLoss
    
class LinearFullyConnectedGPLayer(nn.Module):
    def __init__(self, mt, smt, in_vec_dims, hidden_vec_dims, out_vec_dims):
        super().__init__()
        self.algebra = CliffordAlgebra(mt, split_metric=smt, input_grades=[0,2])
        self.linear_left = MVLinear(self.algebra, in_vec_dims, hidden_vec_dims, subspaces=True, bias=True)
        self.linear_right = MVLinear(self.algebra, in_vec_dims, hidden_vec_dims, subspaces=True, bias=True)
        self.linear_out =  MVLinear(self.algebra, hidden_vec_dims + in_vec_dims, out_vec_dims, subspaces=True, bias=True)
        self.vec_norm = MVLayerNorm(self.algebra, out_vec_dims)

    def forward(self, vec1, vec2=None):
        # normalization
        vec_right = self.linear_right(vec1)
        vec = vec1 if vec2 is None else vec2
        vec_left = self.linear_left(vec)

        # geometric product
        vec_out = self.algebra.geometric_product(vec_left, vec_right)
        vec_out = torch.cat([vec_out, vec1], dim=1)
        vec_out = self.linear_out(vec_out)
        vec_out = self.vec_norm(vec_out) 
        return vec_out


class EGNN_C_Block(nn.Module):
    """ E(n)-equivariant Message Passing Layer """
    def __init__(self, mtr, smtr, node_features_s, hidden_features_s, node_features_v, edge_features, subspaces=False):
        super().__init__()
        self.algebra = CliffordAlgebra(mtr, split_metric=smtr, input_grades=[0,2])
        self.node_features_s = node_features_s
        self.node_features_v = node_features_v
        self.edge_features = edge_features
        self.hidden_features_s = hidden_features_s
        self.factor = self.algebra.n_subspaces if subspaces==True else 1
        self.subspaces = subspaces
        self.message_net = nn.Sequential(nn.Linear(2 * node_features_s + edge_features, hidden_features_s),
                                         nn.ReLU(),
                                         nn.Linear(hidden_features_s, hidden_features_s))
    
        self.update_net = nn.Sequential(nn.Linear(node_features_s + hidden_features_s, hidden_features_s),
                                        nn.ReLU(),
                                        nn.Linear(hidden_features_s, hidden_features_s))
        
        self.pos_net = nn.Sequential(nn.Linear(hidden_features_s, hidden_features_s),
                                     nn.ReLU(),
                                     nn.Linear(hidden_features_s, node_features_v * self.factor))

        self.v_update = LinearFullyConnectedGPLayer(mtr, smtr, node_features_v, node_features_v, node_features_v)
#        self.s_layernorm = nn.LayerNorm(node_features_v * self.factor)
        self.s_layernorm = nn.LayerNorm(hidden_features_s)
        self.v = MVLinear(self.algebra, node_features_v, node_features_v)


    def get_invariants(self, input, algebra):
        norms = algebra.qs(input, grades=algebra.grades[1:]) 
        norms = torch.sqrt(torch.cat(norms, dim=-1) + 1e-8)
        return torch.cat([input[..., :1], norms], dim=-1)

    def forward(self, s, v, edge_index):
        send_idx, rec_idx = edge_index
        s_i, s_j = s[send_idx], s[rec_idx]
        v_i, v_j = v[send_idx], v[rec_idx]  
        message, pos_message = self.message(s_i, s_j, v_i, v_j)
 #       print("shape of message", message.shape)
 #       print("shape of pos_message", pos_message.shape)
        num_messages = torch.bincount(send_idx).unsqueeze(-1)
        message_aggr = global_add_pool(message, rec_idx)
        message_aggr = message_aggr / torch.sqrt(num_messages)
        pos_message_aggr = self.algebra.split(global_add_pool(self.algebra.flatten(pos_message), rec_idx))
        pos_message_aggr = self.algebra.split(self.algebra.flatten(pos_message_aggr) / torch.sqrt(num_messages))
        s, v = self.update(message_aggr, pos_message_aggr, s, v)
        return s, v

    def message(self, s_i, s_j, v_i, v_j):
        """ Create messages """
        v_ij = self.v(v_j - v_i)
        edge_attr = (v_ij * v_ij).sum(dim=-1)
        input = [s_i, s_j, edge_attr]
        input = torch.cat(input, dim=-1)
        message = self.message_net(input)
        pos_message = self.pos_net(message)
        pos_message = v_ij * (pos_message.reshape(pos_message.shape[0], -1, self.algebra.n_subspaces).repeat_interleave(self.algebra.subspaces, dim=-1)) if self.subspaces else v_ij * pos_message.unsqueeze(-1)
        return message, pos_message

    def update(self, message, pos_message, s, v):
        # Update node features
        input = torch.cat((s, message), dim=-1)
        output = self.update_net(input)
        update = s + output
        #use layernorm here
        update = self.s_layernorm(update)
        # Update positions
        v_update = self.v_update(pos_message) + v
        return update, v_update


class RDMFTEGNN_C(nn.Module):
    def __init__(
        self,
#        in_features=[10,25,10], #just an example
        hidden_features=64,
        hidden_features_v=16,
        out_features=1,
        num_layers=3,
        printstep = 1,
        weight_reg=1e-4,
        regtype="L1",
        energyOnly=False,
        nspatialorbs=4
    ):
        super().__init__()
#        self.lossfn = RDM2sLoss(printstep,  energyOnly, weight_reg,regtype,weights=None)
        self.norb = nspatialorbs
        self.split_metric = [self.norb,self.norb]
        
        in_features=[int((self.norb*(self.norb-1))/2),self.norb**2,int((self.norb*(self.norb-1))/2)]
        metric = (1,) * 2*self.norb        

        self.algebra = CliffordAlgebra(metric, split_metric=self.split_metric, input_grades=[0,2])
        self.feature_embedding = MVLinear(self.algebra, sum(in_features), hidden_features_v, subspaces=False, bias=True)
 #       self.sub_feature_embedding_uu = MVLinear(self.algebra, in_features[0], hidden_features_v, subspaces=False, bias=False)
 #       self.sub_feature_embedding_ud = MVLinear(self.algebra, in_features[1], hidden_features_v, subspaces=False, bias=False)
 #       self.sub_feature_embedding_dd = MVLinear(self.algebra, in_features[2], hidden_features_v, subspaces=False, bias=False)
#        self.inv_feature_embedding = nn.Linear(1, hidden_features)
        self.inv_feature_embedding = nn.Linear(1, sum(in_features))
        self.inv_feature_embedding_eigval = nn.Linear(sum(in_features), hidden_features)
        self.input_features = in_features
        layers = []
        for i in range(num_layers):
            layers.append(
#                EGNN_C_Block(hidden_features, hidden_features, hidden_features_v, hidden_features_v, subspaces=True)
                EGNN_C_Block(metric,self.split_metric,sum(in_features), sum(in_features), sum(in_features), sum(in_features), subspaces=True)
            )

#        self.projection_uu =  MVLinear(self.algebra, hidden_features_v, in_features[0], subspaces=False, bias=True)
#        self.projection_ud =  MVLinear(self.algebra, hidden_features_v, in_features[1], subspaces=False, bias=True)
#        self.projection_dd =  MVLinear(self.algebra, hidden_features_v, in_features[2], subspaces=False, bias=True)
        self.projection_uu =  MVLinear(self.algebra, sum(in_features), in_features[0], subspaces=False, bias=True)
        self.projection_ud =  MVLinear(self.algebra, sum(in_features), in_features[1], subspaces=False, bias=True)
        self.projection_dd =  MVLinear(self.algebra, sum(in_features), in_features[2], subspaces=False, bias=True)

        self.projection_suu = nn.Linear(in_features[0], 1)
        self.projection_sud = nn.Linear(in_features[1], 1)
        self.projection_sdd = nn.Linear(in_features[2], 1)
 
    
 
    #       self.projection_logits_uu = MVLinear(self.algebra, hidden_features_v, in_features[0], subspaces=False, bias=True)
 #       self.projection_logits_ud = MVLinear(self.algebra, hidden_features_v, in_features[1], subspaces=False, bias=True)
 #       self.projection_logits_dd = MVLinear(self.algebra, hidden_features_v, in_features[2], subspaces=False, bias=True)
        self.model = nn.Sequential(*layers)

    def _forward(self, s, x, edge_index, batch):
#        x = self.feature_embedding(x)
        for layer in self.model:
            s, x = layer(s, x, edge_index)
            
        s_uu, x_uu = self.projectandslice(x,"uu")
        s_ud, x_ud = self.projectandslice(x,"ud")
        s_dd, x_dd = self.projectandslice(x,"dd")

#        _, l_uu = self.projectandslice_forLogits(x,"uu")
#        _, l_ud = self.projectandslice_forLogits(x,"ud")
#        _, l_dd = self.projectandslice_forLogits(x,"dd")
                
#        print("batch", batch)
 
#        exit()
        #pool over nodes
        x_uu_agg = self.algebra.split(global_add_pool(self.algebra.flatten(x_uu), batch), lastdim=x_uu.shape[-1])
        x_ud_agg = self.algebra.split(global_add_pool(self.algebra.flatten(x_ud), batch), lastdim=x_ud.shape[-1])
        x_dd_agg = self.algebra.split(global_add_pool(self.algebra.flatten(x_dd), batch), lastdim=x_dd.shape[-1])
        
        s_uu_agg = global_add_pool(s_uu, batch)
        s_ud_agg = global_add_pool(s_ud, batch)
        s_dd_agg = global_add_pool(s_dd, batch)

#        l_uu_agg = self.algebra.split(global_add_pool(self.algebra.flatten(l_uu), batch), lastdim=l_uu.shape[-1])
#        l_ud_agg = self.algebra.split(global_add_pool(self.algebra.flatten(l_ud), batch), lastdim=l_ud.shape[-1])
#        l_dd_agg = self.algebra.split(global_add_pool(self.algebra.flatten(l_dd), batch), lastdim=l_dd.shape[-1])


        #sum over channels
        y_uu = torch.einsum("bij,bik->bjk", x_uu_agg, x_uu_agg)
        y_ud = torch.einsum("bij,bik->bjk", x_ud_agg, x_ud_agg)
        y_dd = torch.einsum("bij,bik->bjk", x_dd_agg, x_dd_agg)
        
        e_uu = self.projection_suu(s_uu_agg)
        e_ud = self.projection_sud(s_ud_agg)
        e_dd = self.projection_sdd(s_dd_agg)
        
###predict the corresponding masks for y_uu, y_u, y_dd from x_uu_agg, x_ud_agg, x_dd_agg
##        m_uu =  self.projection_logits_uu(x_uu_agg)
##        m_ud =  self.projection_logits_uu(x_ud_agg)
##        m_dd =  self.projection_logits_uu(x_dd_agg)
#        logits_uu = torch.einsum("bij,bik->bjk", l_uu_agg, l_uu_agg)
#        logits_ud = torch.einsum("bij,bik->bjk", l_ud_agg, l_ud_agg)
#        logits_dd = torch.einsum("bij,bik->bjk", l_dd_agg, l_dd_agg)        
        
        #return x_uu, x_ud, x_dd
#        return e_uu+e_ud+e_dd, [y_uu,y_ud,y_dd],[logits_uu,logits_ud,logits_dd]
        return e_uu+e_ud+e_dd, [y_uu,y_ud,y_dd]
    

    def projectandslice(self,mv,block):
#        proj = self.projection(mv)
        if block =="uu":
            return self.algebra.get_grade_sub(self.projection_uu(mv),2, [0,self.input_features[0]])
        elif block == "ud":
            return self.algebra.get_grade_sub(self.projection_ud(mv),2,[self.input_features[0],self.input_features[0]+self.input_features[1]])
        else:
            return self.algebra.get_grade_sub(self.projection_dd(mv),2,
                                              [self.input_features[0]+self.input_features[1],sum(self.input_features)])

#     def projectandslice_forLogits(self,mv,block):
# #        proj = self.projection(mv)
#         if block =="uu":
#             return self.algebra.get_grade_sub(self.projection_logits_uu(mv),2, [0,self.input_features[0]])
#         elif block == "ud":
#             return self.algebra.get_grade_sub(self.projection_logits_ud(mv),2,[self.input_features[0],self.input_features[0]+self.input_features[1]])
#         else:
#             return self.algebra.get_grade_sub(self.projection_logits_dd(mv),2,
#                                               [self.input_features[0]+self.input_features[1],sum(self.input_features)])
                




    def forward(self, batch, step, mode, lossfn, CalcNrep=False):
#        batch = batch.to("cuda")
        batch_size = batch.ptr.shape[0] - 1

#        print("shape of node_embed_eigevec", batch.node_embed_eigevec.shape)
        eigvec_embed = batch.node_embed_eigevec
        edge_index = batch.edge_index
        

#        print("shape of node_embed_eigeval", batch.node_embed_eigeval.shape)
#        eigval_embed = batch.node_embed_eigeval
#        print("shape of noons", batch.noons.shape)
        noons = batch.noons
        
#        print("dim of noons", noons.shape)
#        print("dim of noons", eigval_embed.shape)

        
        s1 =  self.inv_feature_embedding(noons)
#        s2 =  self.inv_feature_embedding_eigval(eigval_embed)
#        s2 = eigval_embed
        #if you do not embed, first try with embedding - keep it simple, skip the next few lines


       
#        s = s1 + s2
        s = s1 
#        print("dims of s :", s.shape)   
        input_multivector = self.algebra.embed_grade_sub(eigvec_embed, 2)
#        e, y, m = self._forward(s, input_multivector, edge_index, batch.batch)
        e, y = self._forward(s, input_multivector, edge_index, batch.batch)
        ##unpack y_uu and y_dd
#        y_uu = utils.expand_symmetric_matrix_to_tensor(y[0], self.split_metric[0])
#        y_dd = utils.expand_symmetric_matrix_to_tensor(y[2], self.split_metric[1])
        y_ud = y[1].reshape(batch_size,self.norb,self.norb,self.norb,self.norb)
#        print(" expanded y_uu requires grad", y_uu.requires_grad)
#        print("shape of y_uu, y_dd and y_ud", y_uu.shape, y_dd.shape, y_ud.shape)
#        exit()
#        m_ud = m[1].reshape(batch_size,self.norb,self.norb,self.norb,self.norb)
        ##pass it to the loss function
        
#        loss, comp, energies = self.lossfn(e.squeeze(1), [y_uu, y_ud, y_dd], m_ud, batch, step)
        loss, comp, energies = lossfn(e.squeeze(1), [y[0], y_ud, y[2]], batch, step, self.split_metric, CalcNrep)
        t = torch.tensor(comp, dtype=torch.float64)
#        print("t requires grad", t.requires_grad)
        return (
             loss.mean(),
             {"loss":  loss,},
             t,
             energies
         )      
    



