from model_LPSFed import *

from test import test_model
from utils.read_subgraph_data import *
from utils.comparison_eigenvalues import comp_rg_eigenvalues, comp_avg_eigenvalues
from utils.fed_data_preprocessing import dataset_preprocessing, testset_preprocessing, get_item_degrees, get_all_degrees, get_pop_index
from utils.params import DATASET, NUM_CLIENTS

import numpy as np
import torch
import random
import pickle
import json

import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'

def train_model(para, add_para, data, path_excel):
    
    # Updated: val_datas is now returned from read_all_data (like test_datas)
    [train_datas, test_datas, val_datas, pre_train_feature_datas, graph_embeddings_1ds, propagation_embeddings, sparse_propagation_matrix, _, eigenvalue_datas, random_data, random_emb] = data
    [GPU_INDEX, _, MODEL, LR, LAMDA, LAYER, EMB_DIM, BATCH_SIZE, TEST_USER_BATCH, N_EPOCH, IF_PRETRAIN, _, TOP_K] = para[0:13]
    [_, _, _, KEEP_PORB, SAMPLE_RATE, GRAPH_CONV, PREDICTION, LOSS_FUNCTION, GENERALIZATION, OPTIMIZATION, IF_TRASFORMATION, ACTIVATION, POOLING, NUM_CLIENTS, USER_NUM] = para[13:]
    # Handle both old and new parameter formats for backward compatibility
    if len(add_para) >= 11:
        [seed, RANDOMGRAPH_TYPE, FED_RATIO, FED_METHOD, GLOBAL_UPDATE_EPOCH, COLD_THRESHOLD, tau1, tau2, w_lambda, freeze_epoch, margin_ratio] = add_para[:11]
        # New parameters: gamma and omega (default values if not provided)
        if len(add_para) >= 13:
            gamma, omega = add_para[11], add_para[12]
        else:
            gamma, omega = 1.0, 0.5  # Default values
    else:
        raise ValueError(f"add_para must have at least 11 elements, got {len(add_para)}")
    
    # Train datas(n_clients) - list of [train_data, train_data_interaction, user_num, item_num]
    # Test datas(n_clients) - list of [test_data]
    # Val datas(n_clients) - list of [val_data] (loaded from read_all_data, like test_datas)
    
    removed_test_datas = dataset_preprocessing(NUM_CLIENTS, train_datas, test_datas)
    removed_val_datas = dataset_preprocessing(NUM_CLIENTS, train_datas, val_datas)
    para_test = testset_preprocessing(NUM_CLIENTS, train_datas, removed_test_datas, TOP_K, TEST_USER_BATCH, KEEP_PORB)
    para_val = testset_preprocessing(NUM_CLIENTS, train_datas, removed_val_datas, TOP_K, TEST_USER_BATCH, KEEP_PORB)
    
    fed_ratio = FED_RATIO
    fed_method = FED_METHOD    
    
    # Load degree information from JSON files
    item_degrees, user_degrees = [], []
    for i in range(NUM_CLIENTS):
        # Try to load from JSON file first (from subgraph)
        degree_file_path = f'../../data/{DATASET}/{NUM_CLIENTS}_clients/degrees_{i}.json'
        if os.path.exists(degree_file_path):
            with open(degree_file_path, 'r') as f:
                degree_data = json.load(f)
                # Convert string keys to int
                user_degree = {int(k): int(v) for k, v in degree_data['user_degrees'].items()}
                item_degree = {int(k): int(v) for k, v in degree_data['item_degrees'].items()}
                print(f"Loaded degree information for client {i+1} from JSON file")
        else:
            # Fallback to computing from training data
            user_degree, item_degree = get_all_degrees(train_datas[i][1])
            print(f"Computed degree information for client {i+1} from training data")
        
        user_degrees.append(user_degree)
        item_degrees.append(item_degree)        
    
    ## paths of data
    rg_kl_divergences_values = comp_rg_eigenvalues(eigenvalue_datas)
    avg_kl_divergences_values = comp_avg_eigenvalues(eigenvalue_datas)
    kl_values = []
        
    if fed_ratio[0] == 'rg':
        if fed_ratio[1] == 'avg':
            kl_values = rg_kl_divergences_values
        
        elif fed_ratio[1] == 'per':
            kl_values = rg_kl_divergences_values
            kl_values = [1 - value for value in kl_values]
            
    elif fed_ratio[0] == 'avg':
        if fed_ratio[1] == 'avg':
            kl_values = avg_kl_divergences_values
        
        elif fed_ratio[1] == 'per':
            kl_values = avg_kl_divergences_values
            kl_values = [1 - value for value in kl_values]
    
    print('Federated Ratio:', fed_ratio)        
    print('KL Divergence Values:', kl_values)
        
    
    n_clients = NUM_CLIENTS
    models, optimizations, n_clients_batches = [], [], []
    losses = [0 for _ in range(n_clients)]
    F1_maxes = [0 for _ in range(n_clients)]
    NDCG_maxes = [0 for _ in range(n_clients)]
    RECALL_maxes = [0 for _ in range(n_clients)]
    
    # Validation metrics
    F1_val_maxes = [0 for _ in range(n_clients)]
    NDCG_val_maxes = [0 for _ in range(n_clients)]
    RECALL_val_maxes = [0 for _ in range(n_clients)]
    
    cuda = torch.device('cuda:{}'.format(GPU_INDEX)) if torch.cuda.is_available() else torch.device('cpu')
    
    ## define the model for n_clients
    ## Split the training samples into batches
    for i in range(n_clients):
        model = LPSFed(n_users=train_datas[i][2], n_items=train_datas[i][3], lr=LR, lamda=LAMDA,
                    emb_dim=EMB_DIM, layer=LAYER, pre_train_latent_factor=pre_train_feature_datas[0],
                    graph_embeddings=graph_embeddings_1ds[i], graph_conv = GRAPH_CONV,
                    prediction = PREDICTION, loss_function=LOSS_FUNCTION,
                    generalization = GENERALIZATION, optimization=OPTIMIZATION,
                    if_pretrain=IF_PRETRAIN, top_k=TOP_K, if_transformation=IF_TRASFORMATION,
                    activation=ACTIVATION, pooling=POOLING, gpu_index=GPU_INDEX,
                    tau1=tau1, tau2=tau2, w_lambda=w_lambda, margin_ratio=margin_ratio)
        # Set gamma and omega for adaptive margin and refined margin
        model.gamma = gamma
        model.omega = omega
        model.to(cuda)
        models.append(model)
        optimizations.append(model.optimizer)

        batches = list(range(0, len(train_datas[i][1]), BATCH_SIZE))
        batches.append(len(train_datas[i][1]))
        n_clients_batches.append(batches)
    
    updated_eigenvalue_datas = [[] for _ in range(n_clients)] + [eigenvalue_datas[-1]]
    
    updated_pooling_weight, updated_pooling_bias = None, None
    updated_prediction_weight, updated_prediction_bias = None, None
    updated_bc_avg_margin = None
    
    pos_i_embeddings, neg_i_embeddings = [], []
    pos_degree, neg_degree = [], []
    
    time_start = time.time()
    print('Start Training')
    
    # Training Iteratively
    for epoch in range(N_EPOCH):            
        for i in range(n_clients):       
            for batch_num in range(len(n_clients_batches[i]) - 1):
                train_batch_data = []
                for sample in range(n_clients_batches[i][batch_num], n_clients_batches[i][batch_num + 1]):
                    (user, pos_item) = train_datas[i][1][sample]
                    sample_num = 0
                    while sample_num < SAMPLE_RATE:
                        neg_item = random.randint(0, train_datas[i][3] - 1)
                        if neg_item not in train_datas[i][0][user]:
                            sample_num += 1
                            train_batch_data.append([user, pos_item, neg_item])
                train_batch_data = torch.tensor(train_batch_data)
                users = train_batch_data[:, 0]
                pos_items = train_batch_data[:, 1]
                neg_items = train_batch_data[:, 2]
                
                user_size = train_datas[i][2]  # n_users
                item_size = train_datas[i][3]  # n_items
                
                users = torch.clamp(users, 0, user_size - 1)
                pos_items = torch.clamp(pos_items, 0, item_size - 1)
                neg_items = torch.clamp(neg_items, 0, item_size - 1)
                
                user_list = users.detach().cpu().tolist()
                pos_item_list = pos_items.detach().cpu().tolist()
                neg_item_list = neg_items.detach().cpu().tolist()
                
                users_pop = [get_pop_index(user_degrees[i].get(user_id, 0)) for user_id in user_list]
                pos_items_pop = [get_pop_index(item_degrees[i].get(item_id, 0)) for item_id in pos_item_list]
                neg_items_pop = [get_pop_index(item_degrees[i].get(item_id, 0)) for item_id in neg_item_list]
                
                users_pop = torch.tensor(users_pop, dtype=torch.long, device=cuda)
                pos_items_pop = torch.tensor(pos_items_pop, dtype=torch.long, device=cuda)
                neg_items_pop = torch.tensor(neg_items_pop, dtype=torch.long, device=cuda)
                
                try:
                    # Eq. 15: Use refined margin if available from server aggregation
                    # refined_margin is computed from updated_bc_avg_margin (Eq. 14)
                    refined_margin_for_batch = None
                    if LOSS_FUNCTION == 'BC' and updated_bc_avg_margin is not None:
                        # Convert to tensor if needed
                        if isinstance(updated_bc_avg_margin, torch.Tensor):
                            refined_margin_for_batch = updated_bc_avg_margin.item()
                        else:
                            refined_margin_for_batch = float(updated_bc_avg_margin)
                    
                    model_output = models[i](users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop, KEEP_PORB, refined_margin=refined_margin_for_batch)
                    
                    # Check output length and unpack accordingly
                    if len(model_output) == 10:
                        pos_scores, neg_scores, avg_embedding, pos_i_embedding, neg_i_embedding, pop_numerator, pop_denominator, main_numerator, main_denominator, Mc_ui = model_output
                    elif len(model_output) == 9:
                        pos_scores, neg_scores, avg_embedding, pos_i_embedding, neg_i_embedding, pop_numerator, pop_denominator, main_numerator, main_denominator = model_output
                        Mc_ui = None
                    elif len(model_output) == 5:
                        # Backward compatibility: old format without BC components
                        pos_scores, neg_scores, avg_embedding, pos_i_embedding, neg_i_embedding = model_output
                        pop_numerator, pop_denominator, main_numerator, main_denominator = None, None, None, None
                        Mc_ui = None
                    else:
                        raise ValueError(f"Unexpected model output length: {len(model_output)}, expected 5, 9, or 10")

                    # Calculate loss based on loss function
                    if models[i].loss_function == 'BPR':
                        losses[i] = models[i].bpr_loss(pos_scores, neg_scores)
                    elif models[i].loss_function == 'CrossEntropy':
                        losses[i] = models[i].cross_entropy_loss(pos_scores, neg_scores)
                    elif models[i].loss_function == 'MSE':
                        losses[i] = models[i].mse_loss(pos_scores, neg_scores)
                    elif models[i].loss_function == 'BC':
                        if pop_numerator is not None and pop_denominator is not None and main_numerator is not None and main_denominator is not None:
                            # BC-Loss: combines popularity-aware loss and main contrastive loss
                            losses[i] = models[i].bc_loss(pop_numerator, pop_denominator, main_numerator, main_denominator)
                        else:
                            # Fallback to BPR if BC components are not available
                            print(f"Warning: BC components not available for client {i+1}, using BPR loss")
                            losses[i] = models[i].bpr_loss(pos_scores, neg_scores)
                    else:
                        # Default to BPR if loss function is not recognized
                        print(f"Warning: Unknown loss function {models[i].loss_function}, using BPR loss")
                        losses[i] = models[i].bpr_loss(pos_scores, neg_scores)
                        
                except Exception as e:
                    print(f"Error in forward pass for client {i+1}, batch {batch_num}: {e}")
                    print(f"Users shape: {users.shape if isinstance(users, torch.Tensor) else 'N/A'}, "
                          f"Pos items shape: {pos_items.shape if isinstance(pos_items, torch.Tensor) else 'N/A'}, "
                          f"Neg items shape: {neg_items.shape if isinstance(neg_items, torch.Tensor) else 'N/A'}")
                    raise

                optimizations[i].zero_grad()
                losses[i].backward()
                optimizations[i].step()
                
                loss_name = models[i].loss_function
                if epoch == (N_EPOCH-1):
                    pos_i_embedding = pos_i_embedding.detach().cpu().numpy()
                    neg_i_embedding = neg_i_embedding.detach().cpu().numpy()
                    pos_i_embeddings.append(pos_i_embedding)
                    neg_i_embeddings.append(neg_i_embedding)
                    
                    pos_item_list = pos_items.detach().cpu().tolist()
                    neg_item_list = neg_items.detach().cpu().tolist()
                     
                    pos_degree.append([item_degrees[i].get(item_id, 0) for item_id in pos_item_list])
                    neg_degree.append([item_degrees[i].get(item_id, 0) for item_id in neg_item_list])

            # Validation evaluation
            if len(removed_val_datas[i]) > 0 and any(len(items) > 0 for items in removed_val_datas[i]):
                F1_val, RECALL_val, NDCG_val = test_model(models[i], para_val[i])
                if F1_val.max() > F1_val_maxes[i]: F1_val_maxes[i] = F1_val.max()
                if NDCG_val.max() > NDCG_val_maxes[i]: NDCG_val_maxes[i] = NDCG_val.max()
                if RECALL_val.max() > RECALL_val_maxes[i]: RECALL_val_maxes[i] = RECALL_val.max()
            else:
                F1_val, RECALL_val, NDCG_val = np.array([0.0]), np.array([0.0]), np.array([0.0])
            
            # Test evaluation
            F1, RECALL, NDCG = test_model(models[i], para_test[i])
            if F1.max() > F1_maxes[i]: F1_maxes[i] = F1.max()
            if NDCG.max() > NDCG_maxes[i]: NDCG_maxes[i] = NDCG.max()
            if RECALL.max() > RECALL_maxes[i]: RECALL_maxes[i] = RECALL.max()
            
            if (epoch + 1) % GLOBAL_UPDATE_EPOCH == 0:
                if len(removed_val_datas[i]) > 0 and any(len(items) > 0 for items in removed_val_datas[i]):
                    print('Client', i+1, ', epochs:', epoch+1, 
                          ', Val F1 - ', np.round(F1_val_maxes[i], 4), 
                          ', Val Recall - ', np.round(RECALL_val_maxes[i], 4), 
                          ', Val NDCG - ', np.round(NDCG_val_maxes[i], 4),
                          ', Test F1 - ', np.round(F1_maxes[i], 4), 
                          ', Test Recall - ', np.round(RECALL_maxes[i], 4), 
                          ', Test NDCG - ', np.round(NDCG_maxes[i], 4))
                else:
                    print('Client', i+1, ', epochs:', epoch+1, 
                          ', Test F1 - ', np.round(F1_maxes[i], 4), 
                          ', Test Recall - ', np.round(RECALL_maxes[i], 4), 
                          ', Test NDCG - ', np.round(NDCG_maxes[i], 4))
                
                # Collect parameters for federated averaging (only at global update epoch)
                # Update eigenvalue data for federated averaging
                for name, param in models[i].named_parameters():
                    device = param.device
                    if 'kernel' in name: 
                        kernel = param.detach().cpu().numpy()    
                        if len(kernel) <= len(eigenvalue_datas[i]):
                            updated_eigenvalue_datas[i] = kernel * eigenvalue_datas[i][:len(kernel)]
                        else:
                            updated_eigenvalue_datas[i] = kernel[:len(eigenvalue_datas[i])] * eigenvalue_datas[i]
                if i == (n_clients-1): print(" ")
                    
        if (epoch+1) % GLOBAL_UPDATE_EPOCH == 0:
            # Reset parameter lists for federated averaging (collect fresh from all clients)
            pooling_weight, pooling_bias = [], []
            prediction_weight, prediction_bias = [], []
            bc_avg_margin = []
            
            # Collect parameters from all clients
            for i in range(n_clients):
                for name, param in models[i].named_parameters():
                    if 'pooling_W.0' in name: pooling_weight.append(param.detach().cpu().numpy())
                    if 'pooling_b.0' in name: pooling_bias.append(param.detach().cpu().numpy())
                    if 'prediction_W.0' in name: prediction_weight.append(param.detach().cpu().numpy())
                    if 'prediction_b.0' in name: prediction_bias.append(param.detach().cpu().numpy())
                    if LOSS_FUNCTION == 'BC':
                        if 'avg_margin' in name: bc_avg_margin.append(param.detach().cpu().numpy())
            
            ## paths of data
            rg_kl_divergences_values = comp_rg_eigenvalues(updated_eigenvalue_datas)
            avg_kl_divergences_values = comp_avg_eigenvalues(updated_eigenvalue_datas)
            kl_values = []
                
            if fed_ratio[0] == 'rg':
                if fed_ratio[1] == 'avg':
                    kl_values = rg_kl_divergences_values
                
                elif fed_ratio[1] == 'per':
                    kl_values = rg_kl_divergences_values
                    kl_values = [1 - value for value in kl_values]
                    
            elif fed_ratio[0] == 'avg':
                if fed_ratio[1] == 'avg':
                    kl_values = avg_kl_divergences_values
                
                elif fed_ratio[1] == 'per':
                    kl_values = avg_kl_divergences_values
                    kl_values = [1 - value for value in kl_values]
            
            print('Federated Ratio:', fed_ratio)        
            print('KL Divergence Values:', kl_values)
                    
            # Pooling Federated Averaging
            if POOLING == 'MLP3' and len(pooling_weight) > 0: 
                updated_pooling_weight = torch.tensor(np.mean(np.array(pooling_weight), axis=0), dtype=torch.float32)
                updated_pooling_bias = torch.tensor(np.mean(np.array(pooling_bias), axis=0), dtype=torch.float32)
        
            # Prediction Federated Averaging
            if PREDICTION == 'MLP3' and len(prediction_weight) > 0:
                updated_prediction_weight = torch.tensor(np.mean(np.array(prediction_weight), axis=0), dtype=torch.float32)
                updated_prediction_bias = torch.tensor(np.mean(np.array(prediction_bias), axis=0), dtype=torch.float32)
                
            if LOSS_FUNCTION == 'BC' and len(bc_avg_margin) > 0:
                updated_bc_avg_margin = torch.tensor(np.mean(np.array(bc_avg_margin), axis=0), dtype=torch.float32)
                
            # Update the pooling and prediction weight and bias
            for i in range(n_clients):
                if i < len(kl_values):
                    update_ratio = kl_values[i]
                else:
                    update_ratio = 0.5  # Default update ratio if kl_values is shorter
                
                for name, param in models[i].named_parameters():
                    device = param.device
                    
                    if 'pooling_W.0' in name and updated_pooling_weight is not None: 
                        origin = param.data
                        total = (1 - update_ratio) * origin.to(device) + update_ratio * updated_pooling_weight.to(device)
                        param.data = total
                        
                    if 'pooling_b.0' in name and updated_pooling_bias is not None: 
                        origin = param.data
                        total = (1 - update_ratio) * origin.to(device) + update_ratio * updated_pooling_bias.to(device)
                        param.data = total
                    
                    if 'prediction_W.0' in name and updated_prediction_weight is not None: 
                        origin = param.data
                        total = (1 - update_ratio) * origin.to(device) + update_ratio * updated_prediction_weight.to(device)
                        param.data = total
                    
                    if 'prediction_b.0' in name and updated_prediction_bias is not None: 
                        origin = param.data
                        total = (1 - update_ratio) * origin.to(device) + update_ratio * updated_prediction_bias.to(device)
                        param.data = total
                    
                    if 'avg_margin' in name and updated_bc_avg_margin is not None:
                        origin = param.data
                        # Eq. 14: Update avg_margin with similarity-based distribution
                        # Mc_{updated} = (M̄ × ρ̄_c) + (Mc × (1 - ρ̄_c))
                        total = (1 - update_ratio) * origin.to(device) + update_ratio * updated_bc_avg_margin.to(device)
                        param.data = total
        
        if epoch == (N_EPOCH-1):
            model_name = 'lpsfed' + loss_name
            
            # Create emb directory if it doesn't exist
            os.makedirs('./emb', exist_ok=True)
            
            with open('./emb/positive_embeddings_'+model_name+'_.pkl', 'wb') as f:
                pickle.dump(pos_i_embeddings, f)
            with open('./emb/negative_embeddings_'+model_name+'_.pkl', 'wb') as f:
                pickle.dump(neg_i_embeddings, f)
            with open('./emb/positive_degrees_'+model_name+'_.pkl', 'wb') as f:
                pickle.dump(pos_degree, f)
            with open('./emb/negative_degrees_'+model_name+'_.pkl', 'wb') as f:
                pickle.dump(neg_degree, f)
        
    print('Training Finished')
    print('Time:', time.time() - time_start)
    
    # Print final validation and test results
    print('\n=== Final Results ===')
    for i in range(n_clients):
        if len(removed_val_datas[i]) > 0 and any(len(items) > 0 for items in removed_val_datas[i]):
            print(f'Client {i+1} - Validation: F1={np.round(F1_val_maxes[i], 4)}, '
                  f'Recall={np.round(RECALL_val_maxes[i], 4)}, NDCG={np.round(NDCG_val_maxes[i], 4)}')
        print(f'Client {i+1} - Test: F1={np.round(F1_maxes[i], 4)}, '
              f'Recall={np.round(RECALL_maxes[i], 4)}, NDCG={np.round(NDCG_maxes[i], 4)}')
        
    return F1_maxes, NDCG_maxes, RECALL_maxes