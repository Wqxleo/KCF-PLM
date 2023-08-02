# -*- coding: utf-8 -*-
import pickle

import torch
import torch.nn as nn
import time

from framework_kg_aspect.models import RGCNModel, RatingEncoder

from .prediction import PredictionLayer
from .fusion import FusionLayer,FusionAtt
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, opt, Net):
        super(Model, self).__init__()
        self.opt = opt
        # if self.opt.ui_merge == 'cat':
        #     if self.opt.r_id_merge == 'cat':
        #         feature_dim = self.opt.id_emb_size  * 2 * self.opt.num_fea+self.opt.ent_node_dim*2
        #         # aspect_feature_dim = self.opt.id_emb_size * 2 * self.opt.num_fea + self.opt.ent_node_dim * 2+self.opt.aspect_emb_size*2
        #     else:
        #         feature_dim = self.opt.id_emb_size * 2
        #         # aspect_feature_dim = self.opt.id_emb_size * 2
        # else:
        #     if self.opt.r_id_merge == 'cat':
        #         feature_dim = self.opt.id_emb_size * (self.opt.num_fea) # 加上知识图的额外维度
        #     else:
        #         feature_dim = self.opt.id_emb_size+self.opt.ent_node_dim

        if (self.opt.aspect_fusion == 'cat'):
            self.opt.feature_dim = self.opt.last_out_dim * 2
        elif (self.opt.aspect_fusion == 'add' or self.opt.aspect_fusion == 'self'):
            self.opt.feature_dim = self.opt.last_out_dim
        self.model_name = self.opt.model

        self.rating_encoder = RatingEncoder(self.opt,Net)
        self.RGCN = RGCNModel(self.opt)
        self.predict_net = PredictionLayer(self.opt)
        self.dropout1 = nn.Dropout(self.opt.drop_out)
        self.dropout2 = nn.Dropout(self.opt.drop_out)

        if (self.opt.ent_loss):
            self.ent_predict = nn.Linear(opt.ent_node_dim,3)
            self.ent_wight = torch.FloatTensor([1,1,1/opt.aspect_max_len])
            self.ent_wight = self.ent_wight.cuda()



    def forward(self, datas):
        """
        :param datas:
        :return:
        """
        # user_reviews, item_reviews, uids, iids,ent_user_ids,ent_item_ids, user_item2id, item_user2id, user_doc, item_doc, aspect_ids, masks, aspect_ent_ids,weights = datas

        aspect_ids, masks,aspect_ent_ids, weights = datas[-4:]
        uids, iids = datas[2],datas[3]
        ent_user_ids, ent_item_ids = datas[4],datas[5]

        entity_nodes_features = self.RGCN.forward()

        # 图神经网络中用户和项目的表示
        ent_user_rep = entity_nodes_features[ent_user_ids].unsqueeze(1)
        ent_item_rep = entity_nodes_features[ent_item_ids].unsqueeze(1)
        ent_aspect_rep = entity_nodes_features[aspect_ent_ids]
        # ent_aspect_rep = self.ent_fc_a(ent_aspect_rep)
        ent_loss = None
        if(self.opt.ent_loss):
            ent_pre_features = torch.cat([ent_user_rep, ent_item_rep, ent_aspect_rep],dim=1)

            ent_logits = self.ent_predict(ent_pre_features)
            ent_label = torch.LongTensor([[0, 1] + [2] * self.opt.aspect_max_len for _ in range(ent_logits.size()[0])])
            ent_label = ent_label.cuda()
            ent_label = ent_label.view(-1)
            ent_logits = ent_logits.view(-1,3)

            ent_loss = nn.functional.cross_entropy(ent_logits,ent_label,self.ent_wight)




        # 单独测试知识图
        # ui_feature = torch.cat((ent_user_rep,ent_item_rep),dim=2).squeeze(1)

        ui_feature = self.rating_encoder(datas[:-4],ent_user_rep,ent_item_rep)


        # ui_feature = self.dropout2(ui_feature)
        rating_output = self.predict_net(ui_feature, uids, iids).squeeze(1)


        # rating_output = self.predict_net(ui_feature_drop, uids, iids).squeeze(1)

        return rating_output,ent_loss



    def load(self, path):
        '''
        加载指定模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, save_dir, name=None, opt=None):
        '''
        保存模型
        '''

        if name is None:
            name = save_dir + self.model_name + '_kg_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = save_dir + self.model_name + '_kg_' + str(name) + '_' + str(opt) + '.pth'
        torch.save(self.state_dict(), name)
        return name