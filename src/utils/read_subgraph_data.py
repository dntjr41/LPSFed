import json
import numpy as np
import random as rd
import networkx as nx
import torch
import os
from .params import USER_NUM, RAND_USER_NUM
from .read_data import propagation_matrix

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def read_data_new(path):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    f.close()
    user_num = len(data)
    total_user_num = USER_NUM
    item_num = 0
    interactions = []
    for user in range(user_num):
        for item in data[user]:
            interactions.append((user, total_user_num+item))
            item_num = max(item, item_num)
    item_num += 1
    rd.shuffle(interactions)
    return(data, interactions, user_num, item_num)

def read_data(path):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    f.close()
    user_num = len(data)
    item_num = 0
    interactions = []
    for user in range(user_num):
        for item in data[user]:
            interactions.append((user, item))
            item_num = max(item, item_num)
    item_num += 1
    rd.shuffle(interactions)
    return(data, interactions, user_num, item_num)   

def read_subgraph(path, real_user_num):
    
    subgraph = nx.read_gml(path)
    
    user_nodes = [node for node, node_data in subgraph.nodes(data=True) if node_data['bipartite'] == 0]
    item_nodes = [node for node, node_data in subgraph.nodes(data=True) if node_data['bipartite'] == 1]
    
    interactions = []
    user_num = len(user_nodes)
    item_num = len(item_nodes)
    
    for user in user_nodes:
        for item in subgraph[user]:
            interactions.append((int(user), int(item)-real_user_num))
    rd.shuffle(interactions)
    
    data = []
    for user in user_nodes:
        data.append([int(item)-real_user_num for item in subgraph[user]])
    
    item_nodes = sorted([int(item)-real_user_num for item in item_nodes])
    
    return data, interactions, user_num, item_num, item_nodes
    

def read_bases(path, fre_u, fre_v):
    with open(path) as f:
        line = f.readline()
        bases = json.loads(line)
    f.close()
    [feat_u, feat_v] = bases
    feat_u = np.array(feat_u)[:, 0: fre_u].astype(np.float32)
    feat_v = np.array(feat_v)[:, 0: fre_v].astype(np.float32)
    return [feat_u, feat_v]

def read_bases1(path, fre, _if_norm = False):
    with open(path) as f:
        line = f.readline()
        bases = json.loads(line)
    f.close()
    if _if_norm:
        for i in range(len(bases)):
            bases[i] = bases[i]/np.sqrt(np.dot(bases[i], bases[i]))
    return np.array(bases)[:, 0: fre].astype(np.float32)

def read_eigen(path):
    with open(path) as f:
        line = f.readline()
        eigen = json.loads(line)
    f.close()
    return np.array(eigen).astype(np.float32)

def read_all_data(all_para, randgraph_type):
    [_, DATASET, MODEL, _, _, _, EMB_DIM, _, _, _, IF_PRETRAIN, TEST_VALIDATION, TOP_K, FREQUENCY_USER, FREQUENCY_ITEM, FREQUENCY, _, _, GRAPH_CONV, _, _, _, _, _, _, _, PROP_DIM, PROP_EMB, IF_NORM, NUM_CLIENTS, USER_NUM] = all_para
    [hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix] = [0, 0, 0, 0]
    
    ## Paths of data
    DIR = '../data/' + DATASET + '/' + str(NUM_CLIENTS) + '_clients/'
    
    train_paths, test_paths, validation_paths = [], [], []
    hypergraph_embeddings_paths, graph_embeddings_1d_paths, pre_train_feature_paths = [], [], []
    eigenvalue_paths = []
    
    for i in range(NUM_CLIENTS):
        train_paths.append(DIR + 'subgraphs' + str(i) + '.gml')
        test_paths.append(DIR + 'test_data' + str(i) + '.json')
        validation_paths.append(DIR + 'val_data' + str(i) + '.json')
        
        hypergraph_embeddings_paths.append(DIR + 'hypergraph_embeddings' + str(i) + '.json')
        graph_embeddings_1d_paths.append(DIR + 'graph_embeddings_1d' + str(i) + '.json')
        pre_train_feature_paths.append(DIR + 'pre_train_feature' + str(EMB_DIM) + '_' + str(i) + '.json')
        eigenvalue_paths.append(DIR + 'eigenvalues_' + str(i) + '.json')

    random_path = DIR + 'random_train_data_' + randgraph_type + '.gml'
    eigenvalue_paths.append(DIR + 'eigenvalues_random_' + randgraph_type + '.json')
    
    train_datas, test_datas, val_datas = [], [], []
    graph_embeddings_1ds, pre_train_feature_datas = [], []
    eigenvalue_datas, graph_embeddings_random = [], []
    
    print('Reading data...')
    for i in range(NUM_CLIENTS):
        [train_data, train_data_interaction, user_num, item_num, item_nodes] = read_subgraph(train_paths[i], USER_NUM)
        train = [train_data, train_data_interaction, user_num, item_num, item_nodes]
        train_datas.append(train)
        
        # Read test data
        test = read_data(test_paths[i])[0]
        test_datas.append(test)
        
        # Read validation data (like test)
        if os.path.exists(validation_paths[i]):
            val = read_data(validation_paths[i])[0]
            val_datas.append(val)
        else:
            # If validation file doesn't exist, use empty list (like test would handle missing file)
            val_datas.append([])
        
        if IF_PRETRAIN:
            try: pre_train_feature = read_bases(pre_train_feature_paths[i], EMB_DIM, EMB_DIM)
            except:
                print('There is no pre-trained latent factors for client', i)
                pre_train_feature = [0, 0]
                IF_PRETRAIN = False
            pre_train_feature_datas.append(pre_train_feature)
        
        if MODEL == 'LGCN':
            if GRAPH_CONV == '1D': graph_embeddings = read_bases1(graph_embeddings_1d_paths[i], FREQUENCY)
            graph_embeddings_1ds.append(graph_embeddings)    
        eigenvalue_datas.append(read_eigen(eigenvalue_paths[i]))
        
    if MODEL == 'LGCN':
        graph_embeddings_random_path = DIR + 'graph_embeddings_1d_random_' + randgraph_type + '.json'
        if GRAPH_CONV == '1D': graph_embeddings = read_bases1(graph_embeddings_random_path, FREQUENCY)
        graph_embeddings_random.append(graph_embeddings)
    random_data = read_subgraph(random_path, RAND_USER_NUM)
    eigenvalue_datas.append(read_eigen(eigenvalue_paths[NUM_CLIENTS]))
        
    return train_datas, test_datas, val_datas, pre_train_feature_datas, graph_embeddings_1ds, propagation_embeddings, sparse_propagation_matrix, IF_PRETRAIN, eigenvalue_datas, random_data, graph_embeddings_random