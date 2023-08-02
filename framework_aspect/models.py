# -*- coding: utf-8 -*-
import json
import math
import pickle

import torch
import torch.nn as nn
import time
import numpy as np

from gensim.models import KeyedVectors
from torch_geometric.nn import RGCNConv

from .prediction import PredictionLayer
from .fusion import FusionLayer
from torch.nn import TransformerEncoderLayer,TransformerEncoder
import torch.nn.functional as F


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class SelfAtt(nn.Module):
    '''
    self attention for interaction
    '''
    def __init__(self, dim, num_heads,trans_layer_num):
        super(SelfAtt, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(dim, num_heads, 128, 0.4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, trans_layer_num)

    def forward(self,fea,mask):
        fea = fea.permute(1, 0, 2)
        out = self.encoder(fea,src_key_padding_mask=mask)
        return out.permute(1, 0, 2)



class Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.tanh = nn.Tanh()

    def forward(self, query, keys, values, mask):
        """
			e = 512, k = num_reviews
			query shape   :   N X query_len X embed_dim   : (nqe)
			keys shape    :   N X key_len X embed_dim     : (nke)
			values shape  :   N X key_len X embed_dim     : (nke)
		"""
        energy = torch.einsum("nqe,nke->nqk", [query, keys]).squeeze()

        if(mask is not None):
            energy = energy.masked_fill(mask==0, float("-1e20")).unsqueeze(1)

        attention = torch.softmax(energy, dim=2)
        # attention = torch.sigmoid(energy)
        output = torch.einsum("nqk,nke->nqe", [attention, values])
        return self.tanh(output), attention

class RatingEncoder(nn.Module):
    def __init__(self, opt, Net):
        super(RatingEncoder,self).__init__()
        self.opt = opt
        self.net = Net(opt)
        self.fusion_net = FusionLayer(opt)
        # self.predict_net = PredictionLayer(opt)
        self.dropout = nn.Dropout(self.opt.drop_out)



    def forward(self,datas):
        user_feature, item_feature = self.net(datas)

        ui_feature_root = self.fusion_net(user_feature, item_feature)


        return ui_feature_root

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)




class AspectEncode(nn.Module):
    def __init__(self,opt):
        super(AspectEncode,self).__init__()
        self.opt = opt
        self.opt.aspect_feature_dim = opt.aspect_emb_size * 2
        self.predict_aspect_net = nn.Linear(self.opt.aspect_feature_dim,
                                            opt.user_aspect_max_len + opt.item_aspect_max_len)
        self.aspect_embedding = nn.Embedding(opt.aspect_num+2, opt.aspect_emb_size)
        self.aspect_type_embedding = nn.Embedding(4, opt.aspect_emb_size)
        # self.pos_encoder = PositionalEncoding(opt.aspect_emb_size, opt.drop_out)

        if (self.opt.aspect_merge == 'add'):
            aspect_dim_size = opt.aspect_emb_size
        elif(self.opt.aspect_merge == 'cat2'):
            aspect_dim_size = opt.aspect_emb_size
        elif (self.opt.aspect_merge == 'cat3'):
            aspect_dim_size = opt.aspect_emb_size * 2


        self.SelfAtt_1 = SelfAtt(aspect_dim_size, opt.num_heads, opt.trans_layer_num)
        # self.SelfAtt_2 = SelfAtt(aspect_emb_size, opt.num_heads, opt.trans_layer_num)
        self.asp_fc = nn.Linear(aspect_dim_size,opt.id_emb_size*2)
        self.attention_1 = Attention(aspect_dim_size)
        self.dropout1 = nn.Dropout(self.opt.drop_out)
        self.dropout2 = nn.Dropout(self.opt.drop_out)
        self.reset_para()

        # self.attention_2 = Attention(self.opt.aspect_emb_size)
        # self.w = nn.Parameter(torch.ones(3))

    def forward(self,aspect_ids, masks, weights,ui_feature):

        aspect_type = aspect_ids[:,1,:]
        aspect_ids = aspect_ids[:,0,:]
        aspect_feature = self.aspect_embedding(aspect_ids)
        aspect_type_feature = self.aspect_type_embedding(aspect_type)
        if(self.opt.aspect_merge == 'add'):
            aspect_feature = aspect_feature+aspect_type_feature
        elif(self.opt.aspect_merge == 'cat2'):
            aspect_feature = aspect_feature + aspect_type_feature
            # aspect_feature = torch.cat((aspect_feature,aspect_type_feature),dim=2)



        weights = weights.unsqueeze(dim=2)
        aspect_feature = aspect_feature * weights
        # aspect_feature = self.dropout1(aspect_feature)
        aspect_output_1 = self.SelfAtt_1(aspect_feature, masks)
        # aspect_output_2 = self.SelfAtt_2(aspect_feature,masks)

        if(self.opt.query_grad):
            ui_query1 = ui_feature.unsqueeze(dim=1)
            # ui_query2 = ui_feature.unsqueeze(dim=1)
        else:
            with torch.no_grad():
                ui_query1 = ui_feature.unsqueeze(dim=1)
            # ui_query2 = ui_feature.unsqueeze(dim=1)

        aspect_output_1 = self.dropout2(aspect_output_1)
        aspect_output_1 = self.asp_fc(aspect_output_1)
        aspect_output_1, aspect_weight_1 = self.attention_1(ui_query1, aspect_output_1, aspect_output_1, masks)
        # aspect_output_2, aspect_weight_2 = self.attention_2(ui_query1, aspect_output_2, aspect_output_2, masks)

        # aspect_fea = aspect_output.squeeze(dim=1)
        # aspect_pre = self.predict_aspect_net(aspect_fea)
        aspect_output_1 = aspect_output_1.squeeze(dim=1)
        aspect_weight = aspect_weight_1.squeeze(dim=1)

        # aspect_output_1 = self.dropout(aspect_output_1)

        # aspect_output = aspect_output[:, -1, :]
        # aspect_fea = torch.cat((ui_feature_root,aspect_output),dim=1)
        # aspect_pre = self.predict_aspect_net(aspect_fea)

        return aspect_output_1,aspect_weight
    def reset_para(self):
        nn.init.uniform_(self.aspect_type_embedding.weight, -0.2, 0.2)
        if(not self.opt.use_asp_vec):
            nn.init.uniform_(self.aspect_embedding.weight,-0.2,0.2)
        else:
            wv_model = KeyedVectors.load(self.opt.aspect_word2vector)
            embedding = wv_model.wv
            # construct a dict from vocab's index to embedding's index
            embed_dict = {}  # {word:index,...}
            for index, word in enumerate(embedding.index_to_key):
                embed_dict[word] = index

            self.aspect_vectors = np.zeros((self.opt.aspect_num+2, self.opt.aspect_emb_size),dtype=np.float32)

            trained_aspects = []
            untrained_aspects = []
            aspect2id = json.load(open(self.opt.aspect2id,'r'))
            aspect_set = aspect2id.keys()
            for a in aspect_set:
                a_words = a.split()
                flag = False
                vectors = []
                for w in a_words:
                    if(w in embed_dict):
                        vectors.append(embedding[embed_dict[w]])
                        flag = True

                if flag:
                    trained_aspects.append(a)
                    v = np.mean(vectors,axis=0)
                    self.aspect_vectors[aspect2id[a]] = v
                else:
                    untrained_aspects.append(a)

            print('trained aspects: %d, untrained aspects: %d' % (len(trained_aspects), len(untrained_aspects)))
            w2v = torch.from_numpy(self.aspect_vectors)
            self.aspect_embedding.weight.data.copy_(w2v.cuda())




class Model(nn.Module):

    def __init__(self, opt, Net):
        super(Model, self).__init__()
        self.opt = opt
        if self.opt.ui_merge == 'cat':
            if self.opt.r_id_merge == 'cat':
                feature_dim = self.opt.id_emb_size  * 2 * self.opt.num_fea+self.opt.ent_node_dim*2
                # aspect_feature_dim = self.opt.id_emb_size * 2 * self.opt.num_fea + self.opt.ent_node_dim * 2+self.opt.aspect_emb_size*2
            else:
                feature_dim = self.opt.id_emb_size * 2
                # aspect_feature_dim = self.opt.id_emb_size * 2
        else:
            if self.opt.r_id_merge == 'cat':
                feature_dim = self.opt.id_emb_size * (self.opt.num_fea+1) # 加上知识图的额外维度
            else:
                feature_dim = self.opt.id_emb_size+self.opt.ent_node_dim
        if (self.opt.aspect_fusion == 'cat'):
            self.opt.feature_dim = feature_dim*2
        elif (self.opt.aspect_fusion == 'add'):
            self.opt.feature_dim = feature_dim
        self.model_name = self.opt.model

        self.rating_encoder = RatingEncoder(self.opt,Net)
        self.aspect_encoder = AspectEncode(self.opt)
        self.predict_net = PredictionLayer(self.opt)
        self.dropout1 = nn.Dropout(self.opt.drop_out)
        self.dropout2 = nn.Dropout(self.opt.drop_out)
        self.ent_fc_u = nn.Linear(opt.ent_node_dim,self.opt.fc_dim)
        self.ent_fc_i = nn.Linear(opt.ent_node_dim,self.opt.fc_dim)
        # self.ent_fc_a = nn.Linear(opt.ent_node_dim,self.opt.aspect_emb_size)

        # self.w = nn.Parameter(torch.ones(2))

        # self.predict_aspect_net = nn.Linear(self.opt.feature_dim,self.opt.user_aspect_max_len)

    def forward(self, datas):
        """

        :param datas:
        :return:
        """

        # user_reviews, item_reviews, uids, iids,ent_user_ids,ent_item_ids, user_item2id, item_user2id, user_doc, item_doc, aspect_ids, masks, aspect_ent_ids,weights = datas

        aspect_ids, masks,aspect_ent_ids, weights = datas[-4:]
        uids, iids = datas[2],datas[3]
        ent_user_ids, ent_item_ids = datas[4],datas[5]


        # 单独测试知识图
        # ui_feature = torch.cat((ent_user_rep,ent_item_rep),dim=2).squeeze(1)

        ui_feature = self.rating_encoder(datas[:-4])

        ui_feature_drop = self.dropout1(ui_feature)


        aspect_output,aspect_weight = self.aspect_encoder(aspect_ids, masks, weights,ui_feature_drop)
        # aspect_output,aspect_weight = self.aspect_encoder(aspect_ids, masks, weights,ui_feature,ent_aspect_rep)

        """
        log：当aspect_output的权重越小时，aspect的预测效果越好
        """

        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))

        rating_ui_feature = ui_feature_drop+aspect_output.squeeze(1)
        # rating_ui_feature = w1*ui_feature+w2*aspect_output.squeeze(1)

        rating_ui_feature = self.dropout2(rating_ui_feature)
        rating_output = self.predict_net(rating_ui_feature, uids, iids).squeeze(1)

        return rating_output,aspect_weight

    def load(self, path):
        '''
        加载指定模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, epoch=None, name=None, opt=None):
        '''
        保存模型
        '''
        prefix = 'checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_kg_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + self.model_name  + str(name) + '_' + str(opt) + '.pth'
        torch.save(self.state_dict(), name)
        return name


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        # if weights.shape.__len__() == 2 or \
        #         (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
        #     weights = weights.unsqueeze(-1)
        #
        # assert weights.shape.__len__() == loss.shape.__len__()

        losses = loss * weights
        # loss_sum = losses.sum(dim=1)
        # loss_avg = loss_sum.mean()
        loss_sum = losses.sum()
        return loss_sum


