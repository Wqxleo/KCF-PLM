# -*- coding: utf-8 -*-
# author： Wqxiu
# datetime： 2021/11/10 19:50 
# ide： PyCharm

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import BertModel, AutoModel


class BERT(nn.Module):
    '''
    deep conn 2017
    '''
    def __init__(self, opt, uori='user'):
        super(BERT, self).__init__()
        self.opt = opt
        self.num_fea = 2 # DOC,ids

        if(self.opt.bert_type=="bert"):
            self.u_bert = BertModel.from_pretrained(opt.bert_path)
            self.i_bert = BertModel.from_pretrained(opt.bert_path)
            hidden_size=768
        elif(self.opt.bert_type=="tiny_bert"):
            self.u_bert = AutoModel.from_pretrained("prajjwal1/bert-small")
            self.i_bert = AutoModel.from_pretrained("prajjwal1/bert-small")
            hidden_size = 512


        # self.bert = BertModel.from_pretrained(opt.bert_path)
        self.word_embedding = self.u_bert.get_input_embeddings()
        self.word_embedding2 = self.i_bert.get_input_embeddings()
        # unfreeze_layers = ['layer.8', 'layer.9', 'layer.10', 'layer.11', 'pooler.', 'out.']
        # unfreeze_layers = ['layer.10','layer.11','pooler.', 'out.']
        # unfreeze_layers = []

        if(self.opt.freeze):
            # for name, param in self.bert.named_parameters():
            #     param.requires_grad = False
            for name, param in self.u_bert.named_parameters():
                param.requires_grad = False
            for name, param in self.i_bert.named_parameters():
                param.requires_grad = False
        else:
            # for name, param in self.bert.named_parameters():
            #     param.requires_grad = True
            for name, param in self.u_bert.named_parameters():
                param.requires_grad = True
            for name, param in self.i_bert.named_parameters():
                param.requires_grad = True

        self.user_fc = nn.Sequential(
            nn.Linear(hidden_size,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, opt.fc_dim)
        )

        self.item_fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, opt.fc_dim),
        )
        if(self.opt.use_id_emb):
            self.uid_embedding = nn.Embedding(opt.user_num + 2, opt.id_emb_size)
            self.iid_embedding = nn.Embedding(opt.item_num + 2, opt.id_emb_size)

        self.extend_ent_dim = nn.Linear(opt.ent_node_dim,hidden_size)
        self.dropout = nn.Dropout(opt.drop_out)
        self.dropout1 = nn.Dropout(opt.drop_out)
        self.dropout2 = nn.Dropout(opt.drop_out)

        self.reset_para()


    def forward(self, datas,ent_user_rep,ent_item_rep):
        _, _,uids, iids,_,_, _, _, user_doc, item_doc = datas

        user_doc_inputs = user_doc[:, 0, :]
        user_doc_maks = user_doc[:, 1, :].float()
        user_doc_position_ids =user_doc[:, 2, :]

        item_doc_inputs = item_doc[:, 0, :]
        item_doc_maks = item_doc[:, 0, :].float()
        item_doc_position_ids = item_doc[:, 2, :]

        user_doc_inputs_emb = self.word_embedding(user_doc_inputs)
        item_doc_inputs_emb = self.word_embedding2(item_doc_inputs)
        ent_user_rep = self.extend_ent_dim(ent_user_rep)
        ent_item_rep = self.extend_ent_dim(ent_item_rep)

        if(self.opt.cross_att):
            item_doc_inputs_emb[:,0,:] = (item_doc_inputs_emb[:,0,:]+ent_user_rep[:,0,:])/2
            user_doc_inputs_emb[:,0,:] = (user_doc_inputs_emb[:,0,:]+ent_item_rep[:,0,:])/2



        if (self.opt.re_pos):
            u_outputs = self.u_bert(attention_mask =user_doc_maks, inputs_embeds=user_doc_inputs_emb , position_ids=user_doc_position_ids)
            i_outputs = self.i_bert(attention_mask =item_doc_maks, inputs_embeds=item_doc_inputs_emb ,position_ids =item_doc_position_ids)
        else:
            u_outputs = self.u_bert(attention_mask =user_doc_maks, inputs_embeds=user_doc_inputs_emb )
            i_outputs = self.i_bert(attention_mask =item_doc_maks, inputs_embeds=item_doc_inputs_emb)



        user_pool = u_outputs[1]
        item_pool = i_outputs[1]


        user_doc_fea = self.user_fc(user_pool)
        item_doc_fea = self.item_fc(item_pool)


        user_fea = torch.stack([user_doc_fea], 1)
        item_fea = torch.stack([item_doc_fea], 1)

        user_fea = self.dropout1(user_fea)
        item_fea = self.dropout2(item_fea)


        if(self.opt.cross_att):
            return item_fea,user_fea
        else:
            return user_fea, item_fea


    def unfreeze(self):
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = True
        for name, param in self.u_bert.named_parameters():
            param.requires_grad = True
        for name, param in self.i_bert.named_parameters():
            param.requires_grad = True
    def reset_para(self):
        for name,param in self.user_fc.named_parameters():
            nn.init.uniform_(param,-0.02, 0.02)
        for name,param in self.item_fc.named_parameters():
            nn.init.uniform_(param,-0.02, 0.02)

        if(self.opt.use_id_emb):
            nn.init.uniform_(self.uid_embedding.weight, -0.1, 0.1)
            nn.init.uniform_(self.iid_embedding.weight, -0.1, 0.1)

        nn.init.uniform_(self.extend_ent_dim.weight, -0.2, 0.2)
