# -*- coding: utf-8 -*-
# author： Wqxiu
# datetime： 2021/11/10 19:50 
# ide： PyCharm

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import BertModel


class BERT(nn.Module):
    '''
    deep conn 2017
    '''
    def __init__(self, opt, uori='user'):
        super(BERT, self).__init__()
        self.opt = opt
        self.num_fea = 2 # DOC,ids

        self.u_bert = BertModel.from_pretrained(opt.bert_path)
        self.i_bert = BertModel.from_pretrained(opt.bert_path)

        unfreeze_layers = ['layer.8', 'layer.9', 'layer.10', 'layer.11', 'pooler.', 'out.']
        # unfreeze_layers = [ 'layer.11', 'pooler.', 'out.']
        if (self.opt.freeze):
            for name, param in self.u_bert.named_parameters():
                param.requires_grad = False
                for ele in unfreeze_layers:
                    if ele in name:
                        param.requires_grad = True
                        break
            for name, param in self.i_bert.named_parameters():
                param.requires_grad = False
                for ele in unfreeze_layers:
                    if ele in name:
                        param.requires_grad = True
                        break
        self.user_fc = nn.Sequential(
            nn.Linear(768, 128),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, opt.fc_dim)
        )

        self.item_fc = nn.Sequential(
            nn.Linear(768, 128),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, opt.fc_dim),
        )
        if (self.opt.use_id_emb):
            self.uid_embedding = nn.Embedding(opt.user_num + 2, opt.id_emb_size)
            self.iid_embedding = nn.Embedding(opt.item_num + 2, opt.id_emb_size)

        if(self.opt.bert_cnn):
            # share
            self.word_cnn = nn.Conv2d(1, 1, (5, 768), padding=(2, 0))
            # document-level cnn
            self.user_doc_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, 768), padding=(1, 0))
            self.item_doc_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, 768), padding=(1, 0))
            # abstract-level cnn
            self.user_abs_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.filters_num))
            self.item_abs_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.filters_num))

            self.unfold = nn.Unfold((3, opt.filters_num), padding=(1, 0))

            # fc layer
            self.user_fc = nn.Linear(opt.filters_num, opt.id_emb_size)
            self.item_fc = nn.Linear(opt.filters_num, opt.id_emb_size)

        self.dropout = nn.Dropout(self.opt.drop_out)
        self.reset_para()


    def forward(self, datas):
        _, _,uids, iids, _, _, user_doc, item_doc = datas

        user_doc_inputs = user_doc[:,0,:]
        user_doc_maks = user_doc[:,1,:].float()

        user_size = user_doc_inputs.size()[1]

        item_doc_inputs = item_doc[:, 0, :]
        item_doc_maks = item_doc[:, 0, :].float()
        u_outputs = self.u_bert(user_doc_inputs,user_doc_maks)

        i_outputs = self.i_bert(item_doc_inputs,item_doc_maks)

        user_word_embs = u_outputs[0]
        user_pool = u_outputs[1]
        item_word_embs = i_outputs[0]
        item_pool = i_outputs[1]



        if (self.opt.bert_cnn):
            # (BS, 100, DOC_LEN, 1)
            user_local_fea = self.local_attention_cnn(user_word_embs, self.user_doc_cnn)
            item_local_fea = self.local_attention_cnn(item_word_embs, self.item_doc_cnn)

            # DOC_LEN * DOC_LEN
            euclidean = (user_local_fea - item_local_fea.permute(0, 1, 3, 2)).pow(2).sum(1).sqrt()
            attention_matrix = 1.0 / (1 + euclidean)
            # (?, DOC_LEN)
            user_attention = attention_matrix.sum(2)
            item_attention = attention_matrix.sum(1)

            # (?, 32)
            user_doc_fea = self.local_pooling_cnn(user_local_fea, user_attention, self.user_abs_cnn, self.user_fc)
            item_doc_fea = self.local_pooling_cnn(item_local_fea, item_attention, self.item_abs_cnn, self.item_fc)
        else:
            user_doc_fea = self.user_fc(user_pool)
            item_doc_fea = self.item_fc(item_pool)

        if (self.opt.use_id_emb):
            uid_emb = self.uid_embedding(uids)
            iid_emb = self.iid_embedding(iids)
            use_fea = torch.stack([user_doc_fea, uid_emb], 1)
            item_fea = torch.stack([item_doc_fea, iid_emb], 1)
        else:
            use_fea = torch.stack([user_doc_fea], 1)
            item_fea = torch.stack([item_doc_fea], 1)

        return use_fea,item_fea
    def local_attention_cnn(self, word_embs, doc_cnn):
        '''
        :Eq1 - Eq7
        '''
        local_att_words = self.word_cnn(word_embs.unsqueeze(1))
        local_word_weight = torch.sigmoid(local_att_words.squeeze(1))
        word_embs = word_embs * local_word_weight
        d_fea = doc_cnn(word_embs.unsqueeze(1))
        return d_fea

    def local_pooling_cnn(self, feature, attention, cnn, fc):
        '''
        :Eq11 - Eq13
        feature: (?, 100, DOC_LEN ,1)
        attention: (?, DOC_LEN)
        '''
        bs, n_filters, doc_len, _ = feature.shape
        feature = feature.permute(0, 3, 2, 1)  # bs * 1 * doc_len * embed
        attention = attention.reshape(bs, 1, doc_len, 1)  # bs * doc
        pools = feature * attention
        pools = self.unfold(pools)
        pools = pools.reshape(bs, 3, n_filters, doc_len)
        pools = pools.sum(dim=1, keepdims=True)  # bs * 1 * n_filters * doc_len
        pools = pools.transpose(2, 3)  # bs * 1 * doc_len * n_filters

        abs_fea = cnn(pools).squeeze(3)  # ? (DOC_LEN-2), 100
        abs_fea = F.avg_pool1d(abs_fea, abs_fea.size(2))  # ? 100
        abs_fea = F.relu(fc(abs_fea.squeeze(2)))  # ? 32

        return abs_fea

    def reset_para(self):

        if (self.opt.bert_cnn):
            cnns = [self.word_cnn, self.user_doc_cnn, self.item_doc_cnn, self.user_abs_cnn, self.item_abs_cnn]
            for cnn in cnns:
                nn.init.xavier_normal_(cnn.weight)
                nn.init.uniform_(cnn.bias, -0.1, 0.1)

            fcs = [self.user_fc, self.item_fc]
            for fc in fcs:
                nn.init.uniform_(fc.weight, -0.1, 0.1)
                nn.init.constant_(fc.bias, 0.1)

        if (self.opt.use_id_emb):
            nn.init.uniform_(self.uid_embedding.weight, -0.5, 0.5)
            nn.init.uniform_(self.iid_embedding.weight, -0.5, 0.5)