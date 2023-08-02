# -*- coding: utf-8 -*-
# author： Wqxiu
# datetime： 2022/1/12 12:55 
# ide： PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import RGCNConv



class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout,dim):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, dim)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, dim))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(
            dim, dim, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            dim, dim, num_relations * 2, num_bases=num_bases)

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type):
        x = self.entity_embedding(entity)
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, edge_type)

        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)

        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))


