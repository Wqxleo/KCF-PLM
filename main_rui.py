# -*- encoding: utf-8 -*-
import time
import random
import math
import fire

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ReviewData,ERDataset,ERDatasetOldTest
from framework_root import Model
import models_root
import config


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, label,aspect,_ = zip(*batch)
    user_reviews= []
    item_reviews= []
    user_ids= []
    item_ids= []
    user_item2ids= []
    item_user2ids= []
    user_docs= []
    item_docs = []
    for d in data:
        # user_reviews, item_reviews, user_id, item_id, ent_user_id, ent_item_id, user_item2id, item_user2id, user_doc, user_doc_mask, item_doc, item_doc_mask, aspect_ids, aspect_mask, aspect_weights, aspect_emt_ids
        user_review, item_review, user_id, item_id,_,_, user_item2id, item_user2id, user_doc, item_doc,_,_,_,_ = d
        user_reviews.append(user_review)
        item_reviews.append(item_review)
        user_ids.append(user_id)
        item_ids.append(item_id)
        user_item2ids.append(user_item2id)
        item_user2ids.append(item_user2id)
        user_docs.append(user_doc)
        item_docs.append(item_doc)
    return [user_reviews, item_reviews, user_ids, item_ids, user_item2ids, item_user2ids, user_docs, item_docs], label

def count_parameters(model):

    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name)

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(**kwargs):

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    opt.with_kg = False
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models_root, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")


    # 3 data
    train_data = ERDataset(opt, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    val_data = ERDataset(opt, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f'train data: {len(train_data)}; test data: {len(val_data)}',flush=True)

    test_data = ERDataset(opt, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f'The model has {count_parameters(model):,} trainable parameters.')
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # training
    print("start training....",flush=True)
    min_loss = 1e+10
    best_res = 1e+10
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()
    patience = 0

    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        total_maeloss = 0.0
        model.train()
        print(f"{now()}  Epoch {epoch}...",flush=True)
        for idx, (train_datas, scores) in enumerate(train_data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)

            # train_datas = tuple(torch.LongTensor(t).cuda() for t in train_datas)
            train_datas = list(map(lambda x: torch.LongTensor(x).cuda(), train_datas))
            # train_datas = unpack_input(opt, train_datas)

            optimizer.zero_grad()
            output = model(train_datas)
            mse_loss = mse_func(output, scores)
            total_loss += mse_loss.item() * len(scores)

            mae_loss = mae_func(output, scores)
            total_maeloss += mae_loss.item()
            smooth_mae_loss = smooth_mae_func(output, scores)
            if opt.loss_method == 'mse':
                loss = mse_loss
            if opt.loss_method == 'rmse':
                loss = torch.sqrt(mse_loss)
            if opt.loss_method == 'mae':
                loss = mae_loss
            if opt.loss_method == 'smooth_mae':
                loss = smooth_mae_loss
            loss.backward()
            optimizer.step()
            if opt.fine_step:
                if idx % opt.print_step == 0 and idx > 0:
                    print("\t{}, {} step finised;".format(now(), idx),flush=True)
                    val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
                    if val_mse < best_res:
                        patience = 0
                        best_res = val_mse
                        name = model.save(name=opt.dataset, opt=opt.print_opt)
                        print("\tmodel save in ", name)
                    else:
                        patience += 1
                        print("patiencd ++", patience)


        scheduler.step()
        mse = total_loss * 1.0 / len(train_data)
        print(f"{now()}  Epoch {epoch} finished",flush=True)
        print(f"\ttrain data: loss:{total_loss:.4f}, mse: {mse:.4f};",flush=True)

        val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)

        if val_mse < best_res:
            best_res = val_mse
            patience = 0
            name = model.save(name=opt.dataset, opt=opt.print_opt)
            print(f"\tmodel save in ", name,flush=True)
        else:
            patience+=1
            print("patiencd ++",patience,flush=True)

        print("*"*30)

        if(patience>=opt.max_patience):
            print("max patience break",flush=True)
            break

    print("----"*20)
    print(f"{now()} {opt.dataset} {opt.print_opt} best_res:  {best_res}")
    print("----"*20)
    print("best test result")
    model.load(name)
    val_loss, val_mse, val_mae = predict(model, test_data_loader, opt)


def test(**kwargs):

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    assert(len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models_root, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load(opt.pth_path)
    print(f"load model: {opt.pth_path}")
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"{now()}: test in the test datset")
    predict_loss, test_mse, test_mae = predict(model, test_data_loader, opt)


def predict(model, data_loader, opt):
    total_loss = 0.0
    total_maeloss = 0.0
    model.eval()
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            # test_data = unpack_input(opt, test_data)
            test_data = tuple(torch.LongTensor(t).cuda() for t in test_data)

            output = model(test_data)
            mse_loss = torch.sum((output-scores)**2)
            total_loss += mse_loss.item()

            mae_loss = torch.sum(abs(output-scores))
            total_maeloss += mae_loss.item()

    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len
    print(f"\tevaluation reslut: mse: {mse:.4f}; rmse: {math.sqrt(mse):.4f}; mae: {mae:.4f};",flush=True)
    model.train()
    return total_loss, mse, mae


# def unpack_input(opt, x):
#
#     uids, iids = list(zip(*x))
#     uids = list(uids)
#     iids = list(iids)
#
#     user_reviews = opt.users_review_list[uids]
#     user_item2id = opt.user2itemid_list[uids]  # 检索出该user对应的item id
#
#     user_doc = opt.user_doc[uids]
#
#     item_reviews = opt.items_review_list[iids]
#     item_user2id = opt.item2userid_list[iids]  # 检索出该item对应的user id
#     item_doc = opt.item_doc[iids]
#
#     data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
#     data = list(map(lambda x: torch.LongTensor(x).cuda(), data))
#     return data


if __name__ == "__main__":
    fire.Fire()
