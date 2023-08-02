# -*- coding: utf-8 -*-

import numpy as np


class DefaultConfig:

    model = 'DeepCoNN'
    dataset = 'Digital_Music_data'
    # -------------base config-----------------------#
    use_gpu = True
    gpu_id = 0
    multi_gpu = False
    gpu_ids = []

    version = ''
    query_grad = False
    awl = True
    bert_cnn = False
    aspect_max_len = 24
    use_aspect_type = True
    use_aspect_weight = True
    user_review_nums = 6
    item_review_nums = 6
    re_pos = False
    ent_loss = True
    bert_type = "bert"


    seed = 2019
    num_epochs = 200
    num_workers = 0
    attention_dim = 128
    optimizer = 'Adam'
    weight_decay = 2e-4  # optimizer rameteri
    lr = 2e-3
    loss_method = 'mse'
    drop_out = 0.5
    aspect_loss_weight = 2


    use_word_embedding = False

    id_emb_size = 32
    query_mlp_size = 128
    fc_dim = 32

    doc_len = 500
    filters_num = 100
    kernel_size = 3

    num_fea = 1  # id feature, review feature, doc feature
    use_review = True
    use_doc = True
    self_att = False

    r_id_merge = 'cat'  # review and ID feature cat/add
    ui_merge = 'cat'  # cat/add/dot
    output = 'fm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    fine_step = False  # save mode in step level, defualt in epoch
    pth_path = ""  # the saved pth path for test
    print_opt = 'default'
    max_patience = 10
    print_step = 200

    def set_path(self, name):
        '''
        specific
        '''
        self.data_root = f'./dataset/{name}'
        prefix = f'{self.data_root}/train'

        self.user_list_path = f'{prefix}/userReview2Index.npy'
        self.item_list_path = f'{prefix}/itemReview2Index.npy'

        self.user2itemid_path = f'{prefix}/user_item2id.npy'
        self.item2userid_path = f'{prefix}/item_user2id.npy'

        self.user_doc_path = f'{prefix}/userDoc2Index.npy'
        self.item_doc_path = f'{prefix}/itemDoc2Index.npy'

        self.w2v_path = f'{prefix}/w2v.npy'
        self.bert_path = f'./bert-base-uncased'

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        print("load npy from dist...")
        # self.users_review_list = np.load(self.user_list_path, encoding='bytes')
        # self.items_review_list = np.load(self.item_list_path, encoding='bytes')
        # self.user2itemid_list = np.load(self.user2itemid_path, encoding='bytes')
        # self.item2userid_list = np.load(self.item2userid_path, encoding='bytes')
        # self.user_doc = np.load(self.user_doc_path, encoding='bytes')
        # self.item_doc = np.load(self.item_doc_path, encoding='bytes')

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


class rating_exp_movie3_Config(DefaultConfig):

    def __init__(self):
        self.set_path('rating_exp_movie3_data')

    # 超参
    version = ''
    query_grad = False
    awl = False
    rating_lr = 6e-5
    aspect_lr = 2e-3
    ent_lr = 4e-5
    ent_loss_weight = 0.2
    aspect_merge = 'cat2'
    aspect_fusion = 'add'
    use_id_emb = True
    cross_att = True

    fc_dim = ent_node_out = 25
    ent_node_dim = 64
    aspect_emb_size = 64
    last_out_dim = 38

    max_patience = 6
    aspect_weight = 1
    aspect_loss_weight = 0
    lr = 6e-5
    gpu_ids = []
    freeze = True
    aspect_fusion_p = 'f'
    self_att = False

    user_num = 17105 + 2
    item_num = 18582 + 2
    aspect_num = 612

    train_data_size = 44793
    val_data_size = 5473
    test_data_size = 5473

    vocab_size = 50002
    word_dim = 300
    # r_max_len = 202
    # u_max_r = 13
    # i_max_r = 24




    batch_size = 64
    fine_step = False
    print_step = 2000

    num_heads = 4
    trans_layer_num = 4
    ak_merge = 'cat'
    r_id_merge = 'add'  # review and ID feature cat/add
    ui_merge = 'cat'  # cat/add/dot
    output = 'fm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    aspect_max_score = 5
    user_aspect_max_len = 24
    item_aspect_max_len = 24
    aspect_max_len = 24
    ## KG
    n_entity = 45218
    n_relation = 10
    num_blocks = 8
    KnowledgeGraph_path = 'dataset/rating_exp_movie3_data/graph_edge_weight_no_brand.pkl'
    aspect_word2vector = 'dataset/rating_exp_movie3_data/word2vector64.model'
    use_asp_vec = False
    aspect2id = 'dataset/rating_exp_movie3_data/aspect2id_used.json'


class rating_exp_cellphone2_Config(DefaultConfig):

    def __init__(self):
        self.set_path('rating_exp_cellphone2_data')

    # 超参
    version = ''
    query_grad = True
    awl = False
    rating_lr = 6e-5
    aspect_lr = 2e-3
    ent_lr = 4e-5
    aspect_merge = 'cat2'
    aspect_fusion = 'add'
    use_id_emb = False
    cross_att = True
    fc_dim = ent_node_out = 25
    ent_node_dim = 64
    aspect_emb_size = 64
    last_out_dim = 38


    max_patience = 6
    aspect_weight = 1
    aspect_loss_weight = 0
    lr = 6e-5
    gpu_ids = []
    freeze = False
    aspect_fusion_p = 'f'
    self_att = False

    user_num = 18006 + 2
    item_num = 15832 + 2
    aspect_num = 955

    train_data_size = 46466
    val_data_size = 5934
    test_data_size = 5934

    vocab_size = 35391
    word_dim = 300




    batch_size = 64
    fine_step = False
    print_step = 2000

    num_heads = 4
    trans_layer_num = 4
    ak_merge = 'cat'
    r_id_merge = 'add'  # review and ID feature cat/add
    ui_merge = 'cat'  # cat/add/dot
    output = 'fm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    aspect_max_score = 5
    user_aspect_max_len = 24
    item_aspect_max_len = 24

    ## KG
    n_entity = 38890
    n_relation = 10
    num_blocks = 8
    # KnowledgeGraph_path = 'dataset/rating_exp_cellphone2_data/graph_edge_weight.pkl'
    KnowledgeGraph_path = 'dataset/rating_exp_cellphone2_data/graph_edge_weight_no_brand.pkl'
    aspect_word2vector = 'dataset/rating_exp_cellphone2_data/word2vector64.model'
    use_asp_vec = False
    aspect2id = 'dataset/rating_exp_cellphone2_data/aspect2id_used.json'


class rating_exp_automotive_Config(DefaultConfig):

    def __init__(self):
        self.set_path('rating_exp_automotive_data')

    # 超参
    version = ''
    query_grad = False
    awl = False
    rating_lr = 6e-5
    aspect_lr = 2e-3
    ent_lr = 4e-5
    ent_loss_weight = 0.2
    aspect_merge = 'cat2'
    aspect_fusion = 'add'
    use_id_emb = True
    cross_att = True
    fc_dim = ent_node_out = 25
    ent_node_dim = 64
    aspect_emb_size = 64
    last_out_dim = 38


    max_patience = 6
    aspect_weight = 1
    aspect_loss_weight = 0
    lr = 6e-5
    gpu_ids = []
    freeze = False
    aspect_fusion_p = 'f'
    self_att = False

    user_num = 15515 + 2
    item_num = 16888 + 2
    aspect_num = 898

    train_data_size = 41095
    val_data_size = 4636
    test_data_size = 4636

    vocab_size = 26366
    word_dim = 300





    batch_size = 64
    fine_step = False
    print_step = 2000

    num_heads = 4
    trans_layer_num = 4
    ak_merge = 'cat'
    r_id_merge = 'add'  # review and ID feature cat/add
    ui_merge = 'cat'  # cat/add/dot
    output = 'fm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    aspect_max_score = 5
    user_aspect_max_len = 24
    item_aspect_max_len = 24

    ## KG
    n_entity = 36909
    n_relation = 10
    num_blocks = 8
    KnowledgeGraph_path = 'dataset/rating_exp_automotive_data/graph_edge_weight.pkl'
    aspect_word2vector = 'dataset/rating_exp_automotive_data/word2vector64.model'
    use_asp_vec = False
    aspect2id = 'dataset/rating_exp_automotive_data/aspect2id_used.json'
class rating_exp_automotive2_Config(DefaultConfig):

    def __init__(self):
        self.set_path('rating_exp_automotive2_data')

    # 超参
    version = ''
    query_grad = False
    awl = False
    rating_lr = 6e-5
    aspect_lr = 2e-3
    ent_lr = 4e-5
    aspect_merge = 'cat2'
    aspect_fusion = 'add'
    id_emb_size = fc_dim = ent_node_out =  25
    ent_node_dim = 512
    aspect_emb_size =64


    max_patience = 6
    aspect_weight = 1
    aspect_loss_weight = 0
    lr = 6e-5
    gpu_ids = []
    freeze = False
    aspect_fusion_p = 'f'
    self_att = False

    user_num = 13099 + 2
    item_num = 16160 + 2
    aspect_num = 871

    train_data_size = 50887
    val_data_size = 6554
    test_data_size = 6554

    vocab_size = 29074
    word_dim = 300





    batch_size = 64
    fine_step = False
    print_step = 2000

    num_heads = 4
    trans_layer_num = 4
    ak_merge = 'cat'
    r_id_merge = 'add'  # review and ID feature cat/add
    ui_merge = 'cat'  # cat/add/dot
    output = 'fm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    aspect_max_score = 3
    user_aspect_max_len = 24
    item_aspect_max_len = 24

    ## KG
    n_entity = 33485
    n_relation = 10
    num_blocks = 8
    KnowledgeGraph_path = 'dataset/rating_exp_automotive2_data/graph_edge.pkl'
    aspect_word2vector = 'dataset/rating_exp_automotive2_data/word2vector64.model'
    use_asp_vec = False
    aspect2id = 'dataset/rating_exp_automotive2_data/aspect2id_used.json'

# class rating_exp_clothing_Config(DefaultConfig):
#
#     def __init__(self):
#         self.set_path('rating_exp_clothing_data')
#
#     # 超参
#     version = ''
#     query_grad = False
#     awl = False
#     rating_lr = 6e-5
#     aspect_lr = 2e-3
#     ent_lr = 4e-5
#     ent_loss_weight = 0.2
#     aspect_merge = 'add'
#     aspect_fusion = 'add'
#     use_id_emb = False
#
#
#     id_emb_size = fc_dim = ent_node_out =  25
#     ent_node_dim = 64
#     aspect_emb_size =64
#
#     max_patience = 5
#     aspect_weight = 1
#     aspect_loss_weight = 0
#     lr = 6e-5
#     gpu_ids = []
#     freeze = True
#     aspect_fusion_p = 'f'
#     self_att = False
#
#     user_num = 16624 + 2
#     item_num = 23216 + 2
#     aspect_num = 796
#
#     train_data_size = 52453
#     val_data_size =  6512
#     test_data_size = 6512
#
#     vocab_size = 31493
#     word_dim = 300
#
#     batch_size = 64
#     fine_step = False
#     print_step = 2000
#
#     num_heads = 4
#     trans_layer_num = 4
#
#     r_id_merge = 'add'  # review and ID feature cat/add
#     ui_merge = 'cat'  # cat/add/dot
#     output = 'fm'  # 'fm', 'lfm', 'other: sum the ui_feature'
#
#     aspect_max_score = 5
#     user_aspect_max_len = 24
#     item_aspect_max_len = 24
#     aspect_max_len = 24
#
#     ## KG
#     n_entity = 45740
#     n_relation = 10
#     num_blocks = 8
#     # KnowledgeGraph_path = 'dataset/rating_exp_clothing_data/graph_edge_weight.pkl'
#     KnowledgeGraph_path = 'dataset/rating_exp_clothing_data/graph_edge_weight_no_brand.pkl'
#     aspect_word2vector = 'dataset/rating_exp_clothing_data/word2vector64.model'
#     use_asp_vec = False
#     aspect2id = 'dataset/rating_exp_clothing_data/aspect2id_used.json'

class rating_exp_clothing_Config(DefaultConfig):

    def __init__(self):
        self.set_path('rating_exp_clothing_data')

    # 超参
    version = ''
    query_grad = False
    awl = False
    rating_lr = 6e-5
    aspect_lr = 2e-3
    ent_lr = 4e-5
    ent_loss_weight = 0.2
    aspect_merge = 'add'
    aspect_fusion = 'add'
    use_id_emb = False
    cross_att= True

    fc_dim = ent_node_out =  25
    ent_node_dim = 64
    aspect_emb_size = 64
    last_out_dim = 38


    max_patience = 5
    aspect_weight = 1
    aspect_loss_weight = 0
    lr = 6e-5
    gpu_ids = []
    freeze = True
    aspect_fusion_p = 'f'
    self_att = False

    user_num = 16624 + 2
    item_num = 23216 + 2
    aspect_num = 796

    train_data_size = 52453
    val_data_size = 6512
    test_data_size = 6512

    vocab_size = 31493
    word_dim = 300

    batch_size = 64
    fine_step = False
    print_step = 2000

    num_heads = 4
    trans_layer_num = 4

    r_id_merge = 'add'  # review and ID feature cat/add
    ui_merge = 'cat'  # cat/add/dot
    output = 'fm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    aspect_max_score = 5
    user_aspect_max_len = 24
    item_aspect_max_len = 24
    aspect_max_len = 24

    ## KG
    n_entity = 45740
    n_relation = 10
    num_blocks = 8
    # KnowledgeGraph_path = 'dataset/rating_exp_clothing_data/graph_edge_weight.pkl'
    KnowledgeGraph_path = 'dataset/rating_exp_clothing_data/graph_edge_weight_no_brand.pkl'
    aspect_word2vector = 'dataset/rating_exp_clothing_data/word2vector64.model'
    use_asp_vec = False
    aspect2id = 'dataset/rating_exp_clothing_data/aspect2id_used.json'


class rating_exp_yelp2_Config(DefaultConfig):

    def __init__(self):
        self.set_path('rating_exp_yelp2_data')

    # 超参
    version = ''
    query_grad = False
    awl = False
    rating_lr = 6e-5
    aspect_lr = 2e-3
    ent_lr = 4e-5
    ent_loss_weight = 0.2
    aspect_merge = 'cat2'
    aspect_fusion = 'add'
    use_id_emb = False
    cross_att = True

    fc_dim = ent_node_out = 25
    ent_node_dim = 64
    aspect_emb_size = 64
    last_out_dim = 38

    max_patience = 5
    aspect_weight = 1
    aspect_loss_weight = 0
    lr = 6e-5
    gpu_ids = []
    freeze = True
    aspect_fusion_p = 'f'
    self_att = False

    user_num = 19066 + 2
    item_num = 23071 + 2
    aspect_num = 1124

    train_data_size = 32386
    val_data_size =  2850
    test_data_size = 2850

    vocab_size = 39032
    word_dim = 300

    batch_size = 64
    fine_step = False
    print_step = 2000

    num_heads = 4
    trans_layer_num = 4

    r_id_merge = 'add'  # review and ID feature cat/add
    ui_merge = 'cat'  # cat/add/dot
    output = 'fm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    aspect_max_score = 5
    user_aspect_max_len = 24
    item_aspect_max_len = 24
    aspect_max_len = 24

    ## KG
    n_entity = 43254
    n_relation = 7
    num_blocks = 8
    KnowledgeGraph_path = 'dataset/rating_exp_yelp2_data/graph_edge_weight.pkl'
    # KnowledgeGraph_path = 'dataset/rating_exp_clothing_data/graph_edge_weight_no_brand.pkl'
    # aspect_word2vector = 'dataset/rating_exp_clothing_data/word2vector64.model'
    use_asp_vec = False
    aspect2id = 'dataset/rating_exp_yelp2_data/aspect2id_used.json'


class rating_exp_toys2_Config(DefaultConfig):

    def __init__(self):
        self.set_path('rating_exp_toys2_data')

    # 超参
    version = ''
    query_grad = False
    awl = False
    rating_lr = 6e-5
    aspect_lr = 2e-3
    ent_lr = 4e-5
    ent_loss_weight = 0.2
    aspect_merge = 'cat2'
    aspect_fusion = 'add'
    use_id_emb = True
    cross_att = True
    # id_emb_size = fc_dim = ent_node_out =  25
    # ent_node_dim = 64
    # aspect_emb_size =64

    fc_dim = ent_node_out = 25
    ent_node_dim = 64
    aspect_emb_size = 64
    last_out_dim = 38


    max_patience = 6
    aspect_weight = 1
    aspect_loss_weight = 0
    lr = 6e-5
    gpu_ids = []
    freeze = False
    aspect_fusion_p = 'f'
    self_att = False

    user_num = 14437 + 2
    item_num = 15651 + 2
    aspect_num = 904

    train_data_size = 49755
    val_data_size = 6528
    test_data_size = 6528

    vocab_size = 39755
    word_dim = 300


    batch_size = 16
    fine_step = False
    print_step = 2000

    num_heads = 4
    trans_layer_num = 4
    ak_merge = 'cat'
    r_id_merge = 'add'  # review and ID feature cat/add
    ui_merge = 'cat'  # cat/add/dot
    output = 'fm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    aspect_max_score = 5
    user_aspect_max_len = 24
    item_aspect_max_len = 24

    ## KG
    n_entity = 33573
    n_relation = 10
    num_blocks = 8
    KnowledgeGraph_path = 'dataset/rating_exp_toys2_data/graph_edge_weight_no_brand.pkl'
    aspect_word2vector = 'dataset/rating_exp_toys2_data/word2vector64.model'
    use_asp_vec = False
    aspect2id = 'dataset/rating_exp_toys2_data/aspect2id_used.json'