from utils.print_save import print_params, save_params, save_value
import random as rd
import time
import numpy as np
from numpy import log2
import torch

def evaluation_F1(order, top_k, positive_item):
    epsilon = 1e-10
    top_k_items = set(order[0: top_k])
    positive_item = set(positive_item)
    precision = len(top_k_items & positive_item) / max(len(top_k_items), epsilon)
    recall = len(top_k_items & positive_item) / max(len(positive_item), epsilon)
    F1 = 2 * precision * recall / max(precision + recall, epsilon)
    return F1

def evaluation_NDCG(order, top_k, positive_item):
    top_k_item = order[0: top_k]
    epsilon = 1e-10
    DCG = 0
    iDCG = 0 
    for i in range(top_k):
        if top_k_item[i] in positive_item:
            DCG += 1 / log2(i + 2)
    for i in range(min(len(positive_item), top_k)):
        iDCG += 1 / log2(i + 2)
    NDCG = DCG / max(iDCG, epsilon)
    return NDCG

def evaluation_recall(order, positive_item):
    epsilon = 1e-10
    recommended_items = set(order)
    positive_item = set(positive_item)
    recall = len(recommended_items & positive_item) / max(len(positive_item), epsilon)
    return recall

def test_one_user(user, top_item, para_test_one_user):
    [test_data, TOP_K] = para_test_one_user
    k_num = len(TOP_K)
    f1 = np.zeros(k_num)
    recall = np.zeros(k_num)
    ndcg = np.zeros(k_num)
    top_item = top_item.tolist()
    for i in range(k_num):
        f1[i] = evaluation_F1(top_item, TOP_K[i], test_data[user])
        recall[i] = evaluation_recall(top_item, test_data[user])
        ndcg[i] = evaluation_NDCG(top_item, TOP_K[i], test_data[user])
    return f1, recall, ndcg

def test_model(model, para_test):
    train_data, test_data, user_num, item_num, TOP_K, TEST_USER_BATCH, KEEP_PORB = para_test
    
    # Convert TOP_K to list if it's a single integer
    if isinstance(TOP_K, int):
        TOP_K_list = [TOP_K]
        max_k = TOP_K
    else:
        TOP_K_list = TOP_K
        max_k = max(TOP_K)
    
    para_test_one_user = [test_data, TOP_K_list]

    user_top_items = torch.zeros((TEST_USER_BATCH, max_k), dtype=torch.int32)
    test_batch = np.random.choice(user_num, TEST_USER_BATCH, replace=False)
    mini_batch_num = 100
    mini_batch_list = list(range(0, TEST_USER_BATCH, mini_batch_num))
    mini_batch_list.append(TEST_USER_BATCH)
    score_min = -1

    for u in range(len(mini_batch_list) - 1):
        u1, u2 = mini_batch_list[u], mini_batch_list[u + 1]
        user_batch = test_batch[u1:u2]
        items_in_train_data = torch.zeros((u2 - u1, item_num), dtype=torch.int32)

        for u_index, user in enumerate(user_batch):
            # Check if user index is within train_data bounds
            if user < len(train_data):
                for item in train_data[user]:
                    if 0 <= item < item_num:
                        items_in_train_data[u_index, item] = score_min
        
        # For test, pos_items and neg_items should be None to use the first branch in forward()
        # popularity indices are required but not used when pos_items/neg_items are None, so use dummy values
        batch_size = len(user_batch)
        users_pop = torch.zeros(batch_size, dtype=torch.long, device=model.device)
        pos_items_pop = torch.zeros(1, dtype=torch.long, device=model.device)  # Dummy, not used
        neg_items_pop = torch.zeros(1, dtype=torch.long, device=model.device)  # Dummy, not used
        
        user_top_items_batch = model(user_batch, None, None, users_pop, pos_items_pop, neg_items_pop, 1)
        
        # Mask out items in train data
        if user_top_items_batch is not None and len(user_top_items_batch) >= 4:
            _, ratings, _, top_items = user_top_items_batch
            # Set scores for train items to minimum (check bounds before indexing)
            _, item_size = ratings.shape
            for u_index, user in enumerate(user_batch):
                # Check if user index is within train_data bounds
                if user < len(train_data):
                    for item in train_data[user]:
                        # Only mask if item index is within valid range
                        if 0 <= item < item_size:
                            ratings[u_index, item] = -1e10  # Large negative value to exclude from top-k
            # Recompute top-k after masking (ensure k doesn't exceed item_size)
            k = min(max_k, item_size)
            top_items = torch.topk(ratings, k=k, dim=1, largest=True, sorted=True).indices
            user_top_items_batch = (user_top_items_batch[0], ratings, user_top_items_batch[2], top_items)
        if user_top_items_batch is not None:
            top_indices = user_top_items_batch[3]
            # Move to CPU if needed (user_top_items is on CPU)
            if top_indices.is_cuda:
                top_indices = top_indices.cpu()
        else:
            k = min(max_k, item_num)
            top_indices = torch.zeros((u2 - u1, k), dtype=torch.int32)
        
        # Handle case where top_indices might be smaller than max_k
        actual_k = top_indices.shape[1]
        if actual_k <= max_k:
            user_top_items[u1:u2, :actual_k] = top_indices
            # If actual_k < max_k, remaining columns stay as zeros (already initialized)
        else:
            user_top_items[u1:u2] = top_indices[:, :max_k]

    result = []
    for u_index, user in enumerate(test_batch):
        # Check if user index is within test_data bounds
        if user < len(test_data) and len(test_data[user]) > 0:
            result.append(test_one_user(user, user_top_items[u_index], para_test_one_user))

    result = np.array(result)
    F1, RECALL, NDCG = np.mean(np.array(result), axis=0)
    return F1, RECALL, NDCG

def test(path_excel_dir, para_name, para, add_para_name, add_para, data, iter_num):
    # Import here to avoid circular import
    from train_model import train_model
    
    print_params(para_name, para)
    print_params(add_para_name, add_para)
    
    path_excel = path_excel_dir + str(int(time.time())) + str(int(rd.uniform(100, 900))) + '.xlsx'
    para[0] = int(para[0])
    f1, ndcg, recall = train_model(para, add_para, data, path_excel)
        
    path = para[1] + '_' + para[2] + '_' + add_para[1] + '_' + add_para[3] + '_'
    with open('result/' + path + 'result.txt', 'a') as f:
        f.write('f1: ' + str(f1) + '\n' + 
                'recall: ' + str(recall) + '\n' +
                'ndcg: ' + str(ndcg) + '\n')