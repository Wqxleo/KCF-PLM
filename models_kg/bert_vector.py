# @Time : 2022/4/10 21:43
# @Author : qxwang

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import BertModel


class BERT_VEC(nn.Module):
    '''
    deep conn 2017
    '''
    def __init__(self, opt, uori='user'):
        super(BERT_VEC, self).__init__()
        self.opt = opt
        self.num_fea = 2 # DOC,ids

        # self.user_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, 768))
        # self.item_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, 768))
        # self.user_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)
        # self.item_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)

        self.user_fc_linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768,256)
        )
        self.item_fc_linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 256)
        )
        self.user_att = AttentionNet(256)
        self.item_att = AttentionNet(256)

        self.user_fc = nn.Sequential(
            nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, opt.fc_dim)
        )

        self.item_fc = nn.Sequential(
            nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, opt.fc_dim),
        )

        if(self.opt.use_id_emb):
            self.uid_embedding = nn.Embedding(opt.user_num + 2, opt.id_emb_size)
            self.iid_embedding = nn.Embedding(opt.item_num + 2, opt.id_emb_size)

        self.dropout = nn.Dropout(self.opt.drop_out)
        self.reset_para()


    def forward(self, datas):
        _, _,uids, iids,_,_, _, _, user_doc, item_doc = datas


        user_reviews = self.user_fc_linear(user_doc)
        item_reviews = self.item_fc_linear(item_doc)

        u_attention = self.user_att(user_reviews)  # attention:[batch*seq*1]
        u_h = torch.mul(u_attention,user_reviews)  # h:[batch*seq*feature]
        user_doc_fea = torch.sum(u_h, 1)  # sum_h:[batch*feature]

        i_attention = self.item_att(item_reviews)  # attention:[batch*seq*1]
        i_h = torch.mul(i_attention, item_reviews)  # h:[batch*seq*feature]
        item_doc_fea = torch.sum(i_h, 1)  # sum_h:[batch*feature]

        user_doc_fea = self.user_fc(user_doc_fea)
        item_doc_fea = self.item_fc(item_doc_fea)

        # u_fea = F.relu(self.user_cnn(user_doc.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        # i_fea = F.relu(self.item_cnn(item_doc.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        # u_fea = F.avg_pool1d(u_fea, u_fea.size(2)).squeeze(2)
        # i_fea = F.avg_pool1d(i_fea, i_fea.size(2)).squeeze(2)
        # user_doc_fea = self.user_fc_linear(self.dropout(u_fea))
        # item_doc_fea = self.item_fc_linear(self.dropout(i_fea))


        if (self.opt.use_id_emb):
            uid_emb = self.uid_embedding(uids)
            iid_emb = self.iid_embedding(iids)
            use_fea = torch.stack([user_doc_fea, uid_emb], 1)
            item_fea = torch.stack([item_doc_fea, iid_emb], 1)
        else:
            use_fea = torch.stack([user_doc_fea], 1)
            item_fea = torch.stack([item_doc_fea], 1)

        return use_fea, item_fea

    def reset_para(self):
        # for cnn in [self.user_cnn, self.item_cnn]:
        #     nn.init.xavier_normal_(cnn.weight)
        #     nn.init.constant_(cnn.bias, 0.1)
        #
        # for fc in [self.user_fc_linear, self.item_fc_linear]:
        #     nn.init.uniform_(fc.weight, -0.1, 0.1)
        #     nn.init.constant_(fc.bias, 0.1)
        if(self.opt.use_id_emb):
            nn.init.uniform_(self.uid_embedding.weight, -0.5, 0.5)
            nn.init.uniform_(self.iid_embedding.weight, -0.5, 0.5)


class AttentionNet(nn.Module):
    def __init__(self, input_size):
        super(AttentionNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        out = self.linear(x)  # batch*seq*1
        out = F.softmax(out, dim=1)  # batch*seq*1
        return out  # batch*seq*1
