import scipy as sp
import scipy.sparse.linalg
from numpy import *
import numpy as np
import networkx as nx
import random as rd
import json

import torch
import torch.nn.functional as F
import scipy.sparse.linalg as splinalg
import os
from scipy.sparse.linalg import eigsh
import gc

DATASET = 2             # 0 for Amazon, 1 for Gowalla, 2 for ML-1M, 3 for ML-100k, 4 for Yelp2018, 5 for Tmall, 6 for Yelp2022
FREQUENCY = 64          # dimensionality of the base
FREQUENCY_U = [100, 100, 100, 100, 100, 100, 100][DATASET]   # dimensionality of the base of the user graph
FREQUENCY_I = [50, 50, 50, 50, 50, 50, 50][DATASET]    # dimensionality of the base of the user graph
GRAPH_CONV = ['1d', '2d'][0]            # 0 for 1d convolution and 1 for 2d
NUM_CLIENTS = 4
Dataset = ['Amazon', 'Gowalla', 'ML-1M', 'Yelp2018', 'Tmall_buy'][DATASET]
USER_NUM = [52643, 29858, 6022, 31668, 360599][DATASET]
tolerant = 0.1 ** 5
epsilon = 0.1 ** 10
# RANDOMGRAPH_TYPE: 'random' or 'gnmk' (choose one)
RANDOMGRAPH_TYPE = 'gnmk'  # Change to 'random' if needed

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

root = '../../data/'
DIR = root + Dataset + '/' + str(NUM_CLIENTS) + '_clients/'

def read_graph():
    n_clients = NUM_CLIENTS
    
    subgraphs = []
    random_graph = None
    
    for i in range(n_clients):
        subgraph_path = DIR + 'subgraphs' + str(i) + '.gml'
        subgraph = nx.read_gml(subgraph_path)
        user_nodes = [node for node, node_data in subgraph.nodes(data=True) if node_data['bipartite'] == 0]
        item_nodes = [node for node, node_data in subgraph.nodes(data=True) if node_data['bipartite'] == 1]
        
        interactions = []
        user_num = len(user_nodes)
        item_num = len(item_nodes)
        
        for user in user_nodes:
            if user in subgraph:
                for item in subgraph[user]:
                    interactions.append((int(user), int(item)))
        rd.shuffle(interactions)
        subgraphs.append([subgraph, interactions, user_num, item_num])
        print("Subgraph ", i+1,"'s node number", len(subgraph.nodes), ",user number:", user_num, ", item number:", item_num, ", interaction number:", len(interactions))
    
    
    random_graph_data = None
    if RANDOMGRAPH_TYPE:
        random_graph_path = DIR + 'random_train_data_' + RANDOMGRAPH_TYPE + '.gml'
        if os.path.exists(random_graph_path):
            random_graph = nx.read_gml(random_graph_path)
            user_nodes = [node for node, node_data in random_graph.nodes(data=True) if node_data.get('bipartite') == 0]
            item_nodes = [node for node, node_data in random_graph.nodes(data=True) if node_data.get('bipartite') == 1]
            
            interactions = []
            user_num = len(user_nodes)
            item_num = len(item_nodes)
            
            for user in user_nodes:
                if user in random_graph:
                    for item in random_graph[user]:
                        interactions.append((int(user), int(item)))
            rd.shuffle(interactions)
            random_graph_data = [random_graph, interactions, user_num, item_num]
            print(f"Random Graph ({RANDOMGRAPH_TYPE})'s node number", len(random_graph.nodes), 
                  ",user number:", user_num, ", item number:", item_num, 
                  ", interaction number:", len(interactions))
        else:
            print(f"Warning: Random graph file not found: {random_graph_path}")

    return subgraphs, random_graph_data

print('Reading data...')
subgraphs, random_graph_data = read_graph()

def compute_laplacian_matrix_sparse(graph, return_scipy=False):
    """
    Compute Laplacian matrix from graph
    
    Args:
        graph: NetworkX graph
        return_scipy: If True, return scipy sparse matrix; otherwise return PyTorch sparse tensor
    """
    adj_matrix = nx.adjacency_matrix(graph).tocoo()
    degrees = np.array(adj_matrix.sum(1)).flatten()
    degree_matrix = sp.sparse.diags(degrees, format='csr')
    laplacian_matrix = degree_matrix - adj_matrix
    
    if return_scipy:
        return laplacian_matrix.tocsr()
    
    laplacian_matrix = laplacian_matrix.tocoo()
    indices = torch.LongTensor(np.vstack((laplacian_matrix.row, laplacian_matrix.col)))
    values = torch.FloatTensor(laplacian_matrix.data)
    shape = torch.Size(laplacian_matrix.shape)
    laplacian_sparse = torch.sparse_coo_tensor(indices, values, shape).to(DEVICE)
    return laplacian_sparse

def power_iteration_sparse(L, k, num_iter=1000):
    """
    Power iteration with Gram-Schmidt orthogonalization for multiple eigenvectors
    Note: This method may not converge well. Consider using eigenvalue_decomposition_sparse instead.
    """
    n = L.size(0)
    eigenvectors = torch.randn(n, k, device=DEVICE)
    eigenvectors = torch.nn.functional.normalize(eigenvectors, dim=0)
    
    for iter_num in range(num_iter):
        # Apply Laplacian
        new_eigenvectors = torch.sparse.mm(L, eigenvectors)
        
        # Gram-Schmidt orthogonalization
        for j in range(k):
            # Orthogonalize against previous eigenvectors
            for i in range(j):
                proj = torch.sum(new_eigenvectors[:, j] * eigenvectors[:, i])
                new_eigenvectors[:, j] = new_eigenvectors[:, j] - proj * eigenvectors[:, i]
            # Normalize
            norm = torch.norm(new_eigenvectors[:, j])
            if norm > 1e-10:
                new_eigenvectors[:, j] = new_eigenvectors[:, j] / norm
            else:
                new_eigenvectors[:, j] = torch.randn(n, device=DEVICE)
                new_eigenvectors[:, j] = torch.nn.functional.normalize(new_eigenvectors[:, j].unsqueeze(0), dim=1).squeeze(0)
        
        eigenvectors = new_eigenvectors
        torch.cuda.empty_cache()
    
    # Compute eigenvalues using Rayleigh quotient
    eigenvalues = []
    for j in range(k):
        Lv = torch.sparse.mm(L, eigenvectors[:, j].unsqueeze(1)).squeeze(1)
        eigenval = torch.sum(eigenvectors[:, j] * Lv).item()
        eigenvalues.append(eigenval)
    
    return eigenvalues, eigenvectors.cpu().numpy().tolist()

def eigenvalue_decomposition_sparse(L, k):
    """
    Decompose sparse Laplacian matrix using scipy's eigsh
    This is more accurate than power iteration for multiple eigenvectors
    
    Args:
        L: scipy sparse matrix (csr or coo format)
        k: number of eigenvalues/eigenvectors to compute
    
    Returns:
        eigenvalues: list of eigenvalues
        eigenvectors: list of eigenvectors (each column is an eigenvector)
    """
    if isinstance(L, torch.Tensor):
        # Convert PyTorch sparse tensor to scipy sparse matrix
        L = L.cpu()
        indices = L.coalesce().indices().numpy()
        values = L.coalesce().values().numpy()
        shape = L.shape
        L = sp.sparse.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()
    
    # Use eigsh for symmetric matrices (Laplacian is symmetric)
    eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
    # eigenvectors is (n, k) matrix, convert to list of lists (each column is an eigenvector)
    eigenvectors_list = eigenvectors.T.tolist()
    return eigenvalues.tolist(), eigenvectors_list

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
        
def eigenvalue_decomposition_dense(L, k):
    """
    Decompose dense Laplacian matrix using PyTorch
    """
    if isinstance(L, torch.Tensor) and L.is_sparse:
        L_dense = L.to_dense()
    else:
        L_dense = L
    
    eigenvalues, eigenvectors = torch.linalg.eigh(L_dense, UPLO='L')
    # Sort eigenvalues and get top k indices
    sorted_indices = torch.argsort(eigenvalues)
    top_k_indices = sorted_indices[:k]
    # Select top k eigenvalues and eigenvectors
    top_k_values = eigenvalues[top_k_indices]
    top_k_vectors = eigenvectors[:, top_k_indices]
    eigenvalues_list = top_k_values.cpu().detach().numpy().real.tolist()
    eigenvectors_list = top_k_vectors.cpu().detach().numpy().real.tolist()
    return eigenvalues_list, eigenvectors_list

if GRAPH_CONV == '1d':
    print('Initializing...')
    for i in range(NUM_CLIENTS):
        print(f"Processing subgraph {i+1}...")
        # Use scipy sparse matrix for more accurate eigenvalue decomposition
        laplacian_matrix = compute_laplacian_matrix_sparse(subgraphs[i][0], return_scipy=True)
        print('Decomposing the Laplacian matrices...')
        eigenvalues, eigenvectors = eigenvalue_decomposition_sparse(laplacian_matrix, FREQUENCY)
        print(f'Eigenvalues range: min={min(eigenvalues):.6f}, max={max(eigenvalues):.6f}')
        print('Saving features...')
        save_to_json(eigenvectors, DIR + 'graph_embeddings_1d' + str(i) + '.json')
        save_to_json(eigenvalues, DIR + 'eigenvalues_' + str(i) + '.json')
        del laplacian_matrix, eigenvectors, eigenvalues
        torch.cuda.empty_cache()
        gc.collect()
    if random_graph_data is not None and RANDOMGRAPH_TYPE:
        print(f"Processing random graph ({RANDOMGRAPH_TYPE})...")
        # Use scipy sparse matrix for more accurate eigenvalue decomposition
        laplacian_matrix = compute_laplacian_matrix_sparse(random_graph_data[0], return_scipy=True)
        print('Decomposing the Laplacian matrices...')
        eigenvalues, eigenvectors = eigenvalue_decomposition_sparse(laplacian_matrix, FREQUENCY)
        print(f'Eigenvalues range: min={min(eigenvalues):.6f}, max={max(eigenvalues):.6f}')
        print('Saving features...')
        save_to_json(eigenvectors, DIR + 'graph_embeddings_1d_random_' + RANDOMGRAPH_TYPE + '.json')
        save_to_json(eigenvalues, DIR + 'eigenvalues_random_' + RANDOMGRAPH_TYPE + '.json')
        del laplacian_matrix, eigenvectors, eigenvalues
        torch.cuda.empty_cache()
        gc.collect()