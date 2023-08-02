# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionLayer(nn.Module):
    '''
    Fusion Layer for user feature and item feature
    '''
    def __init__(self, opt):
        super(FusionLayer, self).__init__()
        if opt.self_att:
            self.attn = SelfAtt(opt.id_emb_size, opt.num_heads)
        self.opt = opt
        self.linear = nn.Linear(opt.feature_dim, opt.feature_dim)
        self.drop_out = nn.Dropout(0.5)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.constant_(self.linear.bias, 0.1)
        if(self.opt.r_id_merge == 'att'):
            self.user_att = AttentionNet(opt.id_emb_size)
            self.item_att = AttentionNet(opt.id_emb_size)

    def forward(self, u_out, i_out):
        if self.opt.self_att:
            # out = self.attn(u_out, i_out)
            # s_u_out, s_i_out = torch.split(out, out.size(1)//2, 1)
            # u_out = u_out + s_u_out
            # i_out = i_out + s_i_out
            ###############
            u_out = self.attn(u_out)
            i_out = self.attn(i_out)
        if self.opt.r_id_merge == 'cat':
            u_out = u_out.reshape(u_out.size(0), -1)
            i_out = i_out.reshape(i_out.size(0), -1)

        elif(self.opt.r_id_merge == 'add'):
            u_out = u_out.sum(1)
            i_out = i_out.sum(1)
        elif(self.opt.r_id_merge == 'att'):
            u_att = self.user_att(u_out)
            i_att = self.item_att(i_out)

            u_h = torch.mul(u_att, u_out)  # h:[batch*seq*feature]
            u_out = torch.sum(u_h, 1)  # sum_h:[batch*feature]
            i_h = torch.mul(i_att, i_out)  # h:[batch*seq*feature]
            i_out = torch.sum(i_h, 1)  # sum_h:[batch*feature]

        if(self.opt.ui_merge == 'cat'):
            out = torch.cat([u_out, i_out], 1)
        elif self.opt.ui_merge == 'add':
            out = u_out + i_out
        else:
            out = u_out * i_out
        # out = self.drop_out(out)
        # return F.relu(self.linear(out))
        return out


class SelfAtt(nn.Module):
    '''
    self attention for interaction
    '''
    def __init__(self, dim, num_heads):
        super(SelfAtt, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(dim, num_heads, 128, 0.4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 1)

    # def forward(self, user_fea, item_fea):
    #     fea = torch.cat([user_fea, item_fea], 1).permute(1, 0, 2)  # batch * 6 * 64
    #     out = self.encoder(fea)
    #     return out.permute(1, 0, 2)
    def forward(self, fea):
        fea = fea.permute(1, 0, 2)  # batch * 6 * 64
        out = self.encoder(fea)
        return out.permute(1, 0, 2)

class AttentionNet(nn.Module):
    def __init__(self, input_size):
        super(AttentionNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(input_size//2, 1),
        )
    def forward(self, x):
        out = self.linear(x)  # batch*seq*1
        out = F.softmax(out, dim=1)  # batch*seq*1
        return out  # batch*seq*1