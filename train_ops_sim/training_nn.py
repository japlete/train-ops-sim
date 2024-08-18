import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Final, Tuple

class ACTrainNN(nn.Module):
    '''
    Pytorch implementation of the actor-critic network used in the RL algorithm.
    It has embeddings for segment, circuit, train IDs.
    From the train positions it extracts the matrix of differences between pairs of trains.
    From the splines of this positions it extracts the sum per circuit.
    There are multiple layers in common between the actor and critic heads, and also independent layers for each.
    The last layers have skip connections to speed up learning.
    '''
    def __init__(self, n_segments, ntr_per_circ, spline_dim, device, batch_size, embedding_dim : int = 8, hidden_dim : int = 32):
        super(ACTrainNN, self).__init__()

        self.n_trains : Final[int] = sum(ntr_per_circ)
        self.n_circuits : Final[int] = len(ntr_per_circ)
        self.spline_dim : Final[int] = spline_dim
        self.var_idx_st : Final[int] = 11
        tr_circ_end = list(np.cumsum(ntr_per_circ)+self.var_idx_st)
        circ_st_end_idx = list(zip([self.var_idx_st] + tr_circ_end[:-1], tr_circ_end)) # Start and end index for trains in each circuit
        indexing_dvc = 'cpu' # Complex indexing is faster when indexing tensors are on the CPU
        circ_st_idx = torch.LongTensor([t[0] for t in circ_st_end_idx]).requires_grad_(False).to(device) # Start index for each circuit
        self.circ_st_idx : Final[torch.Tensor] = torch.jit.annotate(torch.Tensor, circ_st_idx) 
        circ_st_end_idx = torch.LongTensor(circ_st_end_idx).requires_grad_(False).to(indexing_dvc)
        self.circ_st_end_idx : Final[torch.Tensor] = torch.jit.annotate(torch.Tensor, circ_st_end_idx)
        self.circ_multi_pre : Final[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = tuple([(
            torch.triu_indices(t[1] - t[0], t[1] - t[0], offset=1).requires_grad_(False).to(indexing_dvc), 
            t) for t in circ_st_end_idx if (t[1] - t[0]) > 1]) # Precomputed indices for the pair differences on each circuit with more than 1 train
        circ_st_pairs_idx = torch.triu_indices(self.n_circuits, self.n_circuits, offset=1) # Precomputed indices for the pair differences between circuit heads
        circ_st_pairs_idx = (circ_st_pairs_idx[0] * self.n_circuits + circ_st_pairs_idx[1]).repeat(batch_size)
        circ_st_pairs_idx += (torch.arange(batch_size)*self.n_circuits**2).repeat_interleave(self.n_circuits*(self.n_circuits-1)//2)
        self.circ_st_pairs_idx : Final[torch.Tensor] = torch.jit.annotate(torch.Tensor, circ_st_pairs_idx.requires_grad_(False).to(device))
        spline_circ_st = []
        for i in self.circ_st_idx:
            spline_circ_st += list(range(spline_dim*(i-self.var_idx_st), spline_dim*(i-self.var_idx_st+1))) # Range of spline indices for each circuit head
        self.spline_circ_st_idx : Final[torch.Tensor] = torch.jit.annotate(torch.Tensor, torch.LongTensor(spline_circ_st).requires_grad_(False).to(device))
        self.spline_range : Final[torch.Tensor] = torch.jit.annotate(torch.Tensor, torch.arange(self.spline_dim).repeat(2).reshape(1,-1).requires_grad_(False).to(device))
        self.idx_spl_st : Final[int] = self.var_idx_st + 2*self.n_trains
        
        # Embedding layers
        self.embed_seg = nn.Embedding(num_embeddings= n_segments, embedding_dim= embedding_dim // 2, dtype= torch.float32)
        self.embed_circ = nn.Embedding(num_embeddings= self.n_circuits, embedding_dim= embedding_dim, dtype= torch.float32)
        self.embed_tr = nn.Embedding(num_embeddings= self.n_trains, embedding_dim= embedding_dim, dtype= torch.float32)
        
        # Dense layers
        circ_sizes = torch.LongTensor([t[1] - t[0] for t in circ_st_end_idx])
        self.dense_pairs = nn.Sequential(OrderedDict([
            ('lin_dp', nn.Linear((circ_sizes*(circ_sizes - 1)).sum().item() + self.n_circuits*(self.n_circuits-1)//2, hidden_dim // 2)), ('gelu_dp', nn.GELU())
        ]))
        self.dense_splines = nn.Sequential(OrderedDict([
            ('lin_ds', nn.Linear((2*self.n_circuits + self.n_trains)*spline_dim, hidden_dim // 2)), ('gelu_ds', nn.GELU())
        ]))
        self.dense_pos_waits = nn.Sequential(OrderedDict([
            ('lin_pw', nn.Linear(2*self.n_trains, hidden_dim // 2)), ('gelu_pw', nn.GELU())
        ]))
        self.dense_dec_trains = nn.Sequential(OrderedDict([
            ('lin_dt', nn.Linear(6 + 4*embedding_dim + embedding_dim // 2 + 2*spline_dim, hidden_dim)), ('gelu_dt', nn.GELU())
        ]))
        self.dense_pairs_splines_embeds = nn.Sequential(OrderedDict([
            ('lin_psemb', nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim)), ('gelu_psemb', nn.GELU())
        ]))
        
        # Actor head
        self.dense_all1_actor = nn.Sequential(OrderedDict([
            ('lin_actor1', nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim)), ('gelu_actor1', nn.GELU())
        ]))
        self.dense_all2_actor = nn.Sequential(OrderedDict([
            ('lin_actor2', nn.Linear(hidden_dim, hidden_dim // 2)), ('gelu_actor2', nn.GELU())
        ]))
        self.actor_final = nn.Sequential(OrderedDict([
            ('lin_actor3', nn.Linear(hidden_dim * 4 + 5, hidden_dim // 2)), ('gelu_actor3', nn.GELU()), 
            ('out_actor', nn.Linear(hidden_dim // 2, 1)), ('sigm_actor', nn.Sigmoid())
        ]))

        # Critic head
        self.dense_all1_critic = nn.Sequential(OrderedDict([
            ('lin_critic1', nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim)), ('gelu_critic1', nn.GELU())
        ]))
        self.dense_all2_critic = nn.Sequential(OrderedDict([
            ('lin_critic2', nn.Linear(hidden_dim, hidden_dim // 2)), ('gelu_critic2', nn.GELU())
        ]))
        self.critic_final = nn.Sequential(OrderedDict([
            ('lin_critic3', nn.Linear(hidden_dim * 4, hidden_dim // 2)), ('gelu_critic3', nn.GELU()), 
            ('out_critic', nn.Linear(hidden_dim // 2, 1))
        ]))

    def forward(self, x):
        # Embedding layers
        emb0 = self.embed_seg(x[:, 0].long())
        emb2_3 = self.embed_circ(x[:, 2:4].long()).flatten(start_dim=1)
        dectr_ids = x[:, 4:6].long()
        emb4_5 = self.embed_tr(dectr_ids).flatten(start_dim=1)

        # Pairs difference of train positions and waiting flags for every circuit with more than 1 train
        multi_tr_refs = [(t[0], x[:, t[1][0]:t[1][1]], x[:, t[1][0]+self.n_trains:t[1][1]+self.n_trains])
                         for t in self.circ_multi_pre]
        multi_tr_pairs_cat =  torch.cat([
            (p[1].unsqueeze(1) - p[1].unsqueeze(2))[:, p[0][0], p[0][1]]
            .flatten(start_dim=1) 
            for p in multi_tr_refs
        ] + [
            (p[2].unsqueeze(1) - p[2].unsqueeze(2))[:, p[0][0], p[0][1]]
            .flatten(start_dim=1) 
            for p in multi_tr_refs
        ], dim=1)
        
        # Differences between the first train of each circuit
        circ_heads_pos = torch.index_select(x, 1, self.circ_st_idx)
        circ_heads_pairs_cat = torch.take_along_dim(circ_heads_pos.unsqueeze(1) - circ_heads_pos.unsqueeze(2), 
                                                    self.circ_st_pairs_idx).reshape(x.shape[0], -1) # Indexes are laid out as flattened array
        
        # Pairs dense layer
        pairs_out = self.dense_pairs(torch.cat([multi_tr_pairs_cat, circ_heads_pairs_cat], dim=1)) # sum(n_multi_circ[k]*(n_multi_circ[k]-1)) + n_circ*(n_circ-1)/2 -> 16
        
        # Sum of splines for each circuit
        spline_feats = x[:, self.idx_spl_st:]
        spline_circ_sums = torch.cat([
            spline_feats[:, self.spline_dim*(t[0]-self.var_idx_st) : self.spline_dim*(t[1]-self.var_idx_st)]
             .reshape(-1, t[1]-t[0], self.spline_dim)
             .sum(dim=1) for t in self.circ_st_end_idx], dim=1)

        # Splines of the first train of each circuit
        spline_circ_heads = torch.index_select(spline_feats, 1, self.spline_circ_st_idx)
        
        # Splines dense layer
        splines_out = self.dense_splines(torch.cat([spline_circ_sums, spline_circ_heads, spline_feats], dim=1)) # n_circ*spl + n_circ*spl + n_trains*spl -> 16

        # Decision trains splines
        spline_dec_tr = torch.gather(spline_feats, 1, (dectr_ids*self.spline_dim).repeat_interleave(self.spline_dim,1) + self.spline_range)      
        
        # Decision trains embeddigs, relative positions and splines
        dec_tr_concat = torch.cat([emb0, x[:, (1,)], emb2_3, emb4_5, x[:, 6:11], spline_dec_tr], dim=1) # Dim emb/2+1+2*emb+2*emb+5+2*spl = 6+4.5*emb+2*spl

        # Embeddings and decision trains dense layer
        embeds_dectr_out = self.dense_dec_trains(dec_tr_concat) # 6+5.5*emb+2*spl -> 32

        # Positions, wait flags dense layer
        pos_waits_out = self.dense_pos_waits(x[:, self.var_idx_st:self.idx_spl_st]) # 2*n_trains -> 16

        # Concatenate embeddings, decision trains info, pairs difference output, splines output, positions and waits output and pass through dense layer
        concat_pairs_embeds = torch.cat([embeds_dectr_out, pairs_out, splines_out, pos_waits_out], dim=1) # Dim 32+16+16+16 = 80
        dense_comb_out = self.dense_pairs_splines_embeds(concat_pairs_embeds) # 80 -> 32
        
        # Concatenate output again with features relative to decision trains and splines
        concat_final_common = torch.cat([embeds_dectr_out, splines_out, dense_comb_out], dim=1) # Dim 32+16+32 = 80

        # Actor head splits
        dense_all_1_out_actor = self.dense_all1_actor(concat_final_common) # 80 -> 32
        dense_all_2_out_actor = self.dense_all2_actor(dense_all_1_out_actor) # 32 -> 16
        concat_skips_actor = torch.cat([concat_final_common, dense_all_1_out_actor, dense_all_2_out_actor, x[:, 6:11]], dim=1) # Dim 80+32+16+5 = 133
        out_actor = self.actor_final(concat_skips_actor)

        # Critic head splits
        dense_all_1_out_critic = self.dense_all1_critic(concat_final_common) # 80 -> 32
        dense_all_2_out_critic = self.dense_all2_critic(dense_all_1_out_critic) # 32 -> 16
        concat_skips_critic = torch.cat([concat_final_common, dense_all_1_out_critic, dense_all_2_out_critic], dim=1) # Dim 80+32+16 = 128
        out_critic = self.critic_final(concat_skips_critic)
        
        return out_actor, out_critic

