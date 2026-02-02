import json
import numpy as np
import random as rd
import networkx as nx
import random
import torch
from .params import USER_NUM, COLD_THRESHOLD

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

def read_data_gml(path, json_path):
    graph = nx.read_gml(path)
    
    user_num = 0
    item_num = 0
    interactions = []
    
    for node, node_data in graph.nodes(data=True):
        if 'bipartite' in node_data and node_data['bipartite'] == 0:
            user_num += 1
        elif 'bipartite' in node_data and node_data['bipartite'] == 1:
            item_num = max(node, item_num)
    
    item_num += 1
    
    for edge in graph.edges():
        interactions.append((edge[0], edge[1]))

    rd.shuffle(interactions)
    
    data = []
    for user in range(user_num):
        user_items = [item for item in interactions if item[0] == user]
        data.append(user_items)
    
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    return (data, interactions, user_num, item_num)

def read_data_gml_new(path):
    """
    Read GML file and extract user, item, and interaction data.
    """
    graph = nx.read_gml(path)
    
    user_num = 0
    item_num = 0
    interactions = []

    # Process nodes
    for node, node_data in graph.nodes(data=True):
        if 'bipartite' in node_data and node_data['bipartite'] == 0:  # User nodes
            user_num += 1
        elif 'bipartite' in node_data and node_data['bipartite'] == 1:  # Item nodes
            item_num = max(int(node), item_num)  # Convert node to integer before comparison

    item_num += 1  # Add 1 since indices are zero-based

    # Process edges
    for edge in graph.edges():
        user, item = map(int, edge)  # Ensure both user and item are integers
        interactions.append((user, item))

    rd.shuffle(interactions)  # Shuffle interactions for randomness

    # Group interactions by user
    data = [[] for _ in range(user_num)]
    for user, item in interactions:
        if user < user_num:  # Ensure the user index is within the valid range
            data[user].append(item)

    return data, interactions, user_num, item_num

def split_train_test(self, interactions, test_ratio=0.2):
    num_interactions = len(interactions)
    num_test_interactions = int(num_interactions * test_ratio)
    test_indices = random.sample(range(num_interactions), num_test_interactions)
    
    test_set = [interactions[i] for i in test_indices]
    train_set = [interactions[i] for i in range(num_interactions) if i not in test_indices]
    
    return train_set, test_set
    

def read_dict_data(path):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    user_dict = {user_id: item_list for user_id, item_list in enumerate(data)}
    f.close()
    
    user_num = len(user_dict)
    item_num = max(max(item_list) for item_list in user_dict.values()) + 1
    
    interactions = [(user, item) for user in user_dict for item in user_dict[user]]
    
    rd.shuffle(interactions)
    
    return(user_dict, interactions, user_num, item_num)

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

def read_all_data(all_para):
    [_, DATASET, MODEL, _, _, _, EMB_DIM, _, _, _, IF_PRETRAIN, TEST_VALIDATION, _, FREQUENCY_USER, FREQUENCY_ITEM, FREQUENCY, _, _, GRAPH_CONV, _, _, _, _, _, _, _, PROP_DIM, PROP_EMB, IF_NORM] = all_para
    [hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix] = [0, 0, 0, 0]

    ## Paths of data
    DIR = '../data/' + DATASET + '/'
    # DIR = '../dataset/' + DATASET + '/'
    train_path = DIR + 'train_data.json'
    test_path = DIR + 'test_data.json'
    validation_path = DIR + 'validation_data.json'
    hypergraph_embeddings_path = DIR + 'hypergraph_embeddings.json'                     # hypergraph embeddings
    graph_embeddings_1d_path = DIR + 'graph_embeddings_1d.json'                         # 1d graph embeddings
    graph_embeddings_2d_path = DIR + 'graph_embeddings_2d.json'                         # 2d graph embeddings
    pre_train_feature_path = DIR + 'pre_train_feature' + str(EMB_DIM) + '.json'         # pretrained latent factors

    ## Load data
    ## load training data
    print('Reading data...')
    [train_data, train_data_interaction, user_num, item_num] = read_data(train_path)
    ## load test data
    teat_vali_path = validation_path if TEST_VALIDATION == 'Validation' else test_path
    test_data = read_data(teat_vali_path)[0]
    ## load pre-trained embeddings for all deep models
    if IF_PRETRAIN:
        try: pre_train_feature = read_bases(pre_train_feature_path, EMB_DIM, EMB_DIM)
        except:
            print('There is no pre-trained embeddings found!!')
            pre_train_feature = [0, 0]
            IF_PRETRAIN = False

    ## load pre-trained transform bases for LCFN and SGNN
    if MODEL == 'LGCN':
        if GRAPH_CONV == '1D': graph_embeddings = read_bases1(graph_embeddings_1d_path, FREQUENCY)
        if GRAPH_CONV == '2D_graph': graph_embeddings = read_bases(graph_embeddings_2d_path, FREQUENCY_USER, FREQUENCY_ITEM)
        if GRAPH_CONV == '2D_hyper_graph': graph_embeddings = read_bases(hypergraph_embeddings_path, FREQUENCY_USER, FREQUENCY_ITEM)
        
    return train_data, train_data_interaction, user_num, item_num, test_data, pre_train_feature, hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix, IF_PRETRAIN


def propagation_matrix(graph, user_num, item_num, norm):
    """
    Construct sparse propagation matrix from graph interactions
    
    Args:
        graph: List of (user, item) interactions
        user_num: Number of users
        item_num: Number of items
        norm: Normalization method ('left_norm' or 'sym_norm')
    
    Returns:
        PyTorch sparse tensor representing the propagation matrix
    """
    print('Constructing the sparse graph...')
    eps = 0.1 ** 10
    user_itemNum = np.zeros(user_num)
    item_userNum = np.zeros(item_num)
    
    for (user, item) in graph:
        user_itemNum[user] += 1
        item_userNum[item] += 1
    
    val, idx = [], []
    for (user, item) in graph:
        if norm == 'left_norm':
            val += [1 / max(float(user_itemNum[user]), eps), 1 / max(float(item_userNum[item]), eps)]
            idx += [[user, item + user_num], [item + user_num, user]]
        if norm == 'sym_norm':
            val += [1 / (max(np.sqrt(float(user_itemNum[user] * item_userNum[item])), eps) * 2)]
            idx += [[user, item + user_num], [item + user_num, user]]
    
    # Convert to PyTorch sparse tensor
    indices = torch.LongTensor(idx).t()
    values = torch.FloatTensor(val)
    shape = torch.Size([user_num + item_num, user_num + item_num])
    propagation_matrix_tensor = torch.sparse_coo_tensor(indices, values, shape)
    
    return propagation_matrix_tensor