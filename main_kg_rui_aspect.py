# -*- encoding: utf-8 -*-
import pickle
import time
import random
import math
import fire

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ReviewData,ERDataset
from dataset.data_review import ERDataset4Test
from framework_kg_aspect import Model,SigmoidFocalClassificationLoss,AutomaticWeightedLoss
import models_kg
import config
from sklearn.metrics import classification_report,accuracy_score,precision_score,f1_score,recall_score



def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, scores,aspects,aspect_one_hots = zip(*batch)
    user_reviews= []
    item_reviews= []
    user_ids= []
    item_ids= []
    ent_user_ids = []
    ent_item_ids = []
    user_item2ids= []
    item_user2ids= []
    user_docs= []
    item_docs = []
    aspect_ids = []
    masks = []
    weights = []
    aspect_ent_ids = []

    for d in data:
        user_review, item_review, user_id, item_id,ent_user_id,ent_item_id, user_item2id, item_user2id, user_doc, item_doc,aspect_id,mask,weight,aspect_ent_id = d
        user_reviews.append(user_review)
        item_reviews.append(item_review)
        user_ids.append(user_id)
        item_ids.append(item_id)
        ent_user_ids.append(ent_user_id)
        ent_item_ids.append(ent_item_id)
        user_item2ids.append(user_item2id)
        item_user2ids.append(item_user2id)
        user_docs.append(user_doc)
        item_docs.append(item_doc)
        aspect_ids.append(aspect_id)
        masks.append(mask)
        aspect_ent_ids.append(aspect_ent_id)
        weights.append(weight)
    return [user_reviews, item_reviews, user_ids, item_ids,ent_user_ids,ent_item_ids, user_item2ids, item_user2ids, user_docs, item_docs,aspect_ids,masks,aspect_ent_ids,weights], scores,aspects,aspect_one_hots
def collate_fn4test(batch):
    data, scores,aspects,aspect_one_hots,bucket_types = zip(*batch)
    user_reviews= []
    item_reviews= []
    user_ids= []
    item_ids= []
    ent_user_ids = []
    ent_item_ids = []
    user_item2ids= []
    item_user2ids= []
    user_docs= []
    item_docs = []
    aspect_ids = []
    masks = []
    weights = []
    aspect_ent_ids = []

    for d in data:
        user_review, item_review, user_id, item_id,ent_user_id,ent_item_id, user_item2id, item_user2id, user_doc, item_doc,aspect_id,mask,weight,aspect_ent_id = d
        user_reviews.append(user_review)
        item_reviews.append(item_review)
        user_ids.append(user_id)
        item_ids.append(item_id)
        ent_user_ids.append(ent_user_id)
        ent_item_ids.append(ent_item_id)
        user_item2ids.append(user_item2id)
        item_user2ids.append(item_user2id)
        user_docs.append(user_doc)
        item_docs.append(item_doc)
        aspect_ids.append(aspect_id)
        masks.append(mask)
        aspect_ent_ids.append(aspect_ent_id)
        weights.append(weight)
    return [user_reviews, item_reviews, user_ids, item_ids,ent_user_ids,ent_item_ids, user_item2ids, item_user2ids, user_docs, item_docs,aspect_ids,masks,aspect_ent_ids,weights], scores,aspects,aspect_one_hots,bucket_types
def criterion(y_pred, y_true, weight=None, alpha=0.25, gamma=2):
    sigmoid_p = nn.Sigmoid()(y_pred)
    zeros = torch.zeros_like(sigmoid_p)
    pos_p_sub = torch.where(y_true > zeros,y_true - sigmoid_p,zeros)
    neg_p_sub = torch.where(y_true > zeros,zeros,sigmoid_p)
    per_entry_cross_ent = -alpha * (pos_p_sub ** gamma) * torch.log(torch.clamp(sigmoid_p,1e-8,1.0))-(1-alpha)*(neg_p_sub ** gamma)*torch.log(torch.clamp(1.0-sigmoid_p,1e-8,1.0))

    sum_loss = per_entry_cross_ent.sum(dim=1)
    avg_loss = sum_loss.mean()

    return avg_loss

def count_parameters(model):

    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name)

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metircs(output, aspect_ids,metrics):
    logit = output

    # pred, pred_id = torch.topk(logit, 5, dim=1)  # id=[bs, K]
    # pred = pred.cpu().numpy()
    # pred_id = pred_id.cpu().numpy()

    pred, pred_id = torch.topk(logit, 1, dim=1)  # id=[bs, K]
    for i, gt in enumerate(aspect_ids):
        cand_ids = pred_id[i].tolist()
        if gt in cand_ids:
            metrics['AspectId_Hits@1'] += 1
        metrics['count'] += 1

    pred, pred_id = torch.topk(logit, 3, dim=1)  # id=[bs, K]
    for i, gt in enumerate(aspect_ids):
        cand_ids = pred_id[i].tolist()
        if gt in cand_ids:
            metrics['AspectId_Hits@3'] += 1

    pred, pred_id = torch.topk(logit, 5, dim=1)  # id=[bs, K]
    for i, gt in enumerate(aspect_ids):
        cand_ids = pred_id[i].tolist()
        if gt in cand_ids:
            metrics['AspectId_Hits@5'] += 1


    pass
    # for i, gt in enumerate(y):
    #     gt = gt.item()
    #     cand_ids = pred_id[i].tolist()
    #     if gt in cand_ids:
    #         metrics['TopicId_Hits@5'] += 1

def compute_multilabel_metircs(aspect_num,aspect_pre, aspects_label, aspect_inputs,metrics,aspects_label_ids,aspects_count):
    sigmoid_prob = torch.sigmoid(aspect_pre)
    pred, pred_id = torch.topk(sigmoid_prob, 2, dim=1)  # id=[bs, K]
    pred = pred.tolist()
    pred_id = pred_id.tolist()
    data_len = len(pred_id)
    aspects_pre_ids = []
    index_pres = np.zeros((data_len,aspect_num), int)
    aspect_labels = np.zeros((data_len,aspect_num), int)
    for ind,(index_p,input,label_id) in enumerate(zip(pred_id,aspect_inputs,aspects_label_ids)):
        aspects_pre_id = []
        input = input[0]
        for i in index_p:
            if(input[i] != aspect_num+1 and input[i] != aspect_num):
                aspects_pre_id.append(input[i])

        aspects_pre_ids.append(aspects_pre_id)
        index_pres[ind][aspects_pre_id] = 1
        aspect_labels[ind][label_id] = 1


        # count = aspects_count[aspects_pre_id]
        # aspects_pre_count.append(count)

    metrics['pre'].extend(index_pres)
    metrics['target'].extend(aspect_labels)
    metrics['pre_ids'].extend(aspects_pre_ids)
    metrics['label_ids'].extend(aspects_label_ids)


    pass





def train(**kwargs):

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    opt.with_kg = True
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models_kg, opt.model))


    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            modelP = nn.DataParallel(model, device_ids=opt.gpu_ids)
            model = modelP.module
    if model.rating_encoder.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")


    # 3 data
    train_data = ERDataset(opt, mode="Train")
    aspects_weights,aspects_count = train_data.aspect_label_weight()

    print("all aspects count {}".format(np.sum(aspects_count)))

    # if opt.use_gpu:
    #     aspects_weights = torch.FloatTensor(aspects_weights).cuda()
    # else:
    #     aspects_weights = torch.FloatTensor(aspects_weights)

    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size,shuffle=True, collate_fn=collate_fn)
    val_data = ERDataset(opt, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f'train data: {len(train_data)}; test data: {len(val_data)}')

    test_data = ERDataset(opt, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f'The model has {count_parameters(model):,} trainable parameters.')
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)



    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # training
    print("start training....")


    def loop(best_res,max_patience):
        min_loss = 1e+10
        mse_func = nn.MSELoss()
        bce_func = torch.nn.BCELoss()
        BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()
        sigmoid_focal_loss = SigmoidFocalClassificationLoss()
        mae_func = nn.L1Loss()
        smooth_mae_func = nn.SmoothL1Loss()
        patience = 0
        name = None
        for epoch in range(opt.num_epochs):
            total_loss = 0.0
            total_maeloss = 0.0
            total_entlosses = []
            model.train()
            print(f"{now()}  Epoch {epoch}...",flush=True)
            aspect_losses = []
            for idx, (train_datas, scores, aspects,aspect_label_one_hots) in enumerate(train_data_loader):

                aspects_label = torch.FloatTensor(aspect_label_one_hots)

                if opt.use_gpu:
                    scores = torch.FloatTensor(scores).cuda()
                    aspects_label = aspects_label.cuda()
                else:
                    scores = torch.FloatTensor(scores)
                if (opt.model == "BERT_VEC"):
                    train_datas = tuple([torch.LongTensor(t).cuda() for t in train_datas[:8]]
                                        +[torch.FloatTensor(t).cuda() for t in train_datas[8:10]]
                                        +[torch.LongTensor(t).cuda() for t in train_datas[10:-1]]
                                        +[torch.FloatTensor(train_datas[-1]).cuda()]
                                        )
                else:
                    train_datas = tuple([torch.LongTensor(t).cuda() for t in train_datas[:-1]] +
                                        [torch.FloatTensor(train_datas[-1]).cuda()]
                                        )
                optimizer.zero_grad()
                if len(opt.gpu_ids) > 0:
                    output,aspect_pre,ent_loss = modelP(train_datas)
                else:
                    output, aspect_pre,ent_loss = model(train_datas)

                mse_loss = mse_func(output, scores)

                aspect_loss = criterion(aspect_pre, aspects_label)

                aspect_losses.append(aspect_loss.item())
                total_loss += mse_loss.item() * len(scores)
                if(ent_loss):
                    total_entlosses.append(ent_loss.item())

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

                # if(opt.awl):
                #     loss = awl(loss,opt.aspect_loss_weight*aspect_loss)
                # else:
                #     loss = loss+opt.aspect_loss_weight*aspect_loss
                if(opt.ent_loss):
                    loss += opt.ent_loss_weight*ent_loss
                loss.backward()
                # aspect_loss.backward()
                optimizer.step()


            scheduler.step()
            mse = total_loss * 1.0 / len(train_data)
            print(f"{now()}  Epoch {epoch} finished", flush=True)
            print(f"\ttrain data: loss:{total_loss:.4f}, mse: {mse:.4f};",flush=True)

            aspect_loss_avg = np.mean(aspect_losses)
            print(f"\ttrain data: aspect loss:{aspect_loss_avg:.4f}", flush=True)
            if(opt.ent_loss):
                ent_loss_avg = np.mean(total_entlosses)
                print(f"\ttrain data: ent loss:{ent_loss_avg:.4f}", flush=True)


            subset = 'val'
            val_loss, val_mse, val_mae = predict(model, val_data_loader, opt,aspects_count,subset)
            test_val_loss, test_val_mse, test_val_mae = predict(model, test_data_loader, opt, aspects_count, 'test')

            if val_mse < best_res:
                best_res = val_mse
                patience = 0
                name = model.save('./saved_model/',name=opt.dataset, opt='aspect'+opt.version)
                print("model save in ", name, flush=True)
            else:
                patience+=1
                print("patiencd ++",patience, flush=True)

            print("*"*30)

            if(patience>=max_patience):
                print("max patience break")
                break

        print("----"*20)
        print(f"{now()} {opt.dataset} {opt.print_opt} best_res:  {best_res}", flush=True)
        print("----"*20)

        if(name):
            print("best test result")

            model.load(name)
            subset = 'test'
            val_loss, val_mse, val_mae = predict(model, test_data_loader, opt,aspects_count,subset)
            return best_res

    if(opt.freeze):

        best_res = loop(1e+10,3)
        print("BERT unfreeze!!!!!!!!!!!!!!!!!!!!!!!!!")
        model.rating_encoder.net.unfreeze()
        loop(best_res,opt.max_patience)
    else:
        best_res = loop(1e+10, opt.max_patience)








def test(**kwargs):

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    opt.with_kg = True
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    opt.parse(kwargs)
    assert(len(opt.pth_path) > 0)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)
    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models_kg, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    if model.rating_encoder.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load('./saved_model/'+opt.pth_path)
    print(f"load model: {opt.pth_path}")

    train_data = ERDataset(opt, mode="Train")
    aspects_weights, aspects_count = train_data.aspect_label_weight()

    print("all aspects count {}".format(np.sum(aspects_count)))

    if opt.use_gpu:
        aspects_weights = torch.FloatTensor(aspects_weights).cuda()
    else:
        aspects_weights = torch.FloatTensor(aspects_weights)

    # test_data = ERDataset(opt, mode="Test")
    # test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    # val_loss, val_mse, val_mae = predict(model, test_data_loader, opt, aspects_count, 'test')

    # bucket = [[1,12],[12,24],[24,36],[36,10000],[1,10000]]

    bucket = [[1,10000]]
    total = 0
    for s,e in bucket:
        print('aspects from [{},{})'.format(s,e))
        test_data = ERDataset4Test(opt, mode="Test",aspects_min_len=s,aspects_max_len=e)
        test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn4test)

        print("test data len ", len(test_data))
        total += len(test_data)
        print(f"{now()}: test in the test datset")
        subset = 'test'
        val_loss, val_mse, val_mae = predict4test(model, test_data_loader, opt, aspects_count, subset)
    #
    # print("total len",total)

def predict4test(model, data_loader, opt,aspects_count,subset):
    total_loss = 0.0
    total_maeloss = 0.0
    aspect_losses = []
    bce_func = nn.BCELoss()
    sigmoid_focal_loss = SigmoidFocalClassificationLoss()
    BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()

    model.eval()

    metrics_test = {"pre":[],"target":[],"pre_ids":[],"label_ids":[]}

    # bucket = {0: [1, 100],
    #           1: [100, 200],
    #           2: [200, 300],
    #           }
    # bucket_count = {0: 0,
    #                 1: 0,
    #                 2: 0,
    #                 }
    # bucket_loss = {0: 0,
    #                1: 0,
    #                2: 0,
    #                }
    bucket = {0: [1, 12],
              1: [12, 24],
              2: [24, 36],
              3: [36, 1000],
              }
    bucket_count = {0: 0,
                    1: 0,
                    2: 0,
                    3:0
                    }
    bucket_loss = {0: 0,
                   1: 0,
                   2: 0,
                   3:0
                   }

    with torch.no_grad():
        for idx, (test_data, scores,aspects_label_ids,aspect_label_one_hots,bucket_types) in enumerate(data_loader):
            aspect_inputs = test_data[-4]

            # aspects_label = np.zeros((len(aspects), opt.aspect_num), int)
            # for i, a in enumerate(aspects):
            #     for index in a:
            #         aspects_label[i][index] = 1
            # aspects_label = torch.FloatTensor(aspects_label)
            aspects_label = torch.FloatTensor(aspect_label_one_hots)


            bucket_indexes = {0: [],
                              1: [],
                              2: [],
                              3:[]
                      }

            for i in range(len(bucket_types)):
                for b in bucket_indexes:
                    if(bucket_types[i]==b):
                        bucket_indexes[b].append(i)
                        bucket_count[b] += 1
                        break

            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
                aspects_label = aspects_label.cuda()
            else:
                scores = torch.FloatTensor(scores)

            # test_data = unpack_input(opt, test_data)
            test_data = tuple(
                [torch.LongTensor(t).cuda() for t in test_data[:-1]] + [torch.FloatTensor(test_data[-1]).cuda()])

            output,aspect_pre = model(test_data)
            diff  = (output-scores)**2

            for b in bucket_indexes:
                bucket_loss[b] += torch.sum(diff[bucket_indexes[b]]).item()


            mse_loss = torch.sum(diff)
            # aspect_loss = bce_func(aspect_pre, aspects_label)
            # aspect_loss = sigmoid_focal_loss(aspect_pre, aspects_label, aspects_weights)
            # aspect_loss = BCEWithLogitsLoss(aspect_pre, aspects_label)
            # aspect_loss = bce_func(aspect_pre, aspects_label)
            aspect_loss = criterion(aspect_pre, aspects_label)
            compute_multilabel_metircs(opt.aspect_num,aspect_pre, aspects_label, aspect_inputs,metrics_test,aspects_label_ids,aspects_count)
            aspect_losses.append(aspect_loss.item())
            total_loss += mse_loss.item()
            mae_loss = torch.sum(abs(output-scores))
            total_maeloss += mae_loss.item()

    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len
    print(f"\tevaluation reslut:total loss: {total_loss:.4f}; mse: {mse:.4f}; rmse: {math.sqrt(mse):.4f}; mae: {mae:.4f};",flush=True)
    aspect_loss_avg = np.mean(aspect_losses)

    for b in bucket_loss:
        print('buckets {}-{} mse loss'.format(bucket[b][0],bucket[b][1]))
        print('bucket len ',bucket_count[b])
        b_mse = bucket_loss[b]*1.0/bucket_count[b]
        print(f" mse: {b_mse:.4f}")




    recall = recall_score(metrics_test['target'], metrics_test['pre'],average='micro')
    precision = precision_score(metrics_test['target'], metrics_test['pre'], average='micro')
    f1 = f1_score(metrics_test['target'], metrics_test['pre'], average='micro')

    if(subset == 'test'):
        aspect_pre_saved_path = opt.data_root+'/test/test_aspect_pre_result.pkl'

        pickle.dump(metrics_test['pre_ids'],open(aspect_pre_saved_path,'wb'))

    print(f"\tevaluation reslut:\n \taspect loss: {aspect_loss_avg:.4f},precision: {precision:.4f},f1: {f1:.4f}，recall：{recall:.4f}",flush=True)

    model.train()

    return total_loss, mse, mae

def predict(model, data_loader, opt,aspects_count,subset):
    total_loss = 0.0
    total_maeloss = 0.0
    aspect_losses = []
    bce_func = nn.BCELoss()
    sigmoid_focal_loss = SigmoidFocalClassificationLoss()
    BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()

    model.eval()

    metrics_test = {"pre":[],"target":[],"pre_ids":[],"label_ids":[]}

    with torch.no_grad():
        for idx, (test_data, scores,aspects_label_ids,aspect_label_one_hots) in enumerate(data_loader):
            aspect_inputs = test_data[-4]

            # aspects_label = np.zeros((len(aspects), opt.aspect_num), int)
            # for i, a in enumerate(aspects):
            #     for index in a:
            #         aspects_label[i][index] = 1
            # aspects_label = torch.FloatTensor(aspects_label)
            aspects_label = torch.FloatTensor(aspect_label_one_hots)


            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
                aspects_label = aspects_label.cuda()
            else:
                scores = torch.FloatTensor(scores)

            # test_data = unpack_input(opt, test_data)
            if (opt.model == "BERT_VEC"):
                test_data = tuple([torch.LongTensor(t).cuda() for t in test_data[:8]]
                                    + [torch.FloatTensor(t).cuda() for t in test_data[8:10]]
                                    + [torch.LongTensor(t).cuda() for t in test_data[10:-1]]
                                    +[torch.FloatTensor(test_data[-1]).cuda()]
                                    )
            else:
                test_data = tuple([torch.LongTensor(t).cuda() for t in test_data[:-1]] +
                                    [torch.FloatTensor(test_data[-1]).cuda()]
                                    )

            output,aspect_pre,ent_loss = model(test_data)
            diff  = (output-scores)**2


            mse_loss = torch.sum(diff)
            # aspect_loss = bce_func(aspect_pre, aspects_label)
            # aspect_loss = sigmoid_focal_loss(aspect_pre, aspects_label, aspects_weights)
            # aspect_loss = BCEWithLogitsLoss(aspect_pre, aspects_label)
            # aspect_loss = bce_func(aspect_pre, aspects_label)
            aspect_loss = criterion(aspect_pre, aspects_label)
            compute_multilabel_metircs(opt.aspect_num,aspect_pre, aspects_label, aspect_inputs,metrics_test,aspects_label_ids,aspects_count)
            aspect_losses.append(aspect_loss.item())
            total_loss += mse_loss.item()
            mae_loss = torch.sum(abs(output-scores))
            total_maeloss += mae_loss.item()

    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len
    print("{} result:".format(subset))
    print(f"\tevaluation reslut:total loss: {total_loss:.4f}; mse: {mse:.4f}; rmse: {math.sqrt(mse):.4f}; mae: {mae:.4f};",flush=True)

    aspect_loss_avg = np.mean(aspect_losses)

    recall = recall_score(metrics_test['target'], metrics_test['pre'],average='micro')
    precision = precision_score(metrics_test['target'], metrics_test['pre'], average='micro')
    f1 = f1_score(metrics_test['target'], metrics_test['pre'], average='micro')

    if(subset == 'test'):
        aspect_pre_saved_path = opt.data_root+'/test/test_aspect_pre_result.pkl'

        pickle.dump(metrics_test['pre_ids'],open(aspect_pre_saved_path,'wb'))

    print(f"\tevaluation reslut:\n \taspect loss: {aspect_loss_avg:.4f},precision: {precision:.4f},f1: {f1:.4f}，recall：{recall:.4f}",flush=True)

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
