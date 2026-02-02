import networkx.algorithms.bipartite as bipartite
from sklearn.cluster import SpectralClustering
from collections import Counter
import scipy as sp
import networkx as nx
import os
import random as rd
import json

from params import NUM_CLIENTS, RANDOMGRAPH_TYPE, DATASET, USER_NUM, ITEM_NUM
from read_data import *


class MakeSubgraph():
    # Path constants
    DATA_DIR = '../../data/'
    CLIENTS_DIR_SUFFIX = '_clients'
    SUBGRAPHS_PREFIX = 'subgraphs'
    RANDOM_GRAPH_PREFIX = 'random_train_data'
    
    def __init__(self):
        self.base_dir = self.DATA_DIR + DATASET + '/'
        self.clients_dir = self.base_dir + str(NUM_CLIENTS) + self.CLIENTS_DIR_SUFFIX + '/'
        self.train_data_path = self.base_dir + 'train_data.json'
        self.val_data_path = self.base_dir + 'val_data.json'
        self.test_data_path = self.base_dir + 'test_data.json'
        self.subgraphs_path = self.clients_dir + self.SUBGRAPHS_PREFIX
    
    def generate_random_graph(self):
        _, interactions, user_num, item_num = read_data(self.train_data_path)
        
        rd_num_nodes1 = int(round(user_num / NUM_CLIENTS))
        rd_num_nodes2 = int(round(item_num / NUM_CLIENTS))
        
        if RANDOMGRAPH_TYPE == 'random':
            rd_num_edges = int(round(len(interactions) / NUM_CLIENTS))
        elif RANDOMGRAPH_TYPE == 'gnmk':
            rd_num_edges = int(round(len(interactions) / NUM_CLIENTS) / NUM_CLIENTS)
        
        rd_edge_ratio = rd_num_edges / (rd_num_nodes1 + rd_num_nodes2)**2
        
        G = None
        if RANDOMGRAPH_TYPE == 'random':
            G = bipartite.random_graph(n=rd_num_nodes1, m=rd_num_nodes2, p=rd_edge_ratio)
            print("Random Graph's node number:", len(G.nodes), ", edge number:", len(G.edges))
            save_random_path = self.clients_dir + self.RANDOM_GRAPH_PREFIX + '_random'
        elif RANDOMGRAPH_TYPE == 'gnmk':
            G = bipartite.gnmk_random_graph(n=rd_num_nodes1, m=rd_num_nodes2, k=rd_num_edges)
            print("Random Graph's node number:", len(G.nodes), ", edge number:", len(G.edges))
            save_random_path = self.clients_dir + self.RANDOM_GRAPH_PREFIX + '_gnmk'
        
        if G is not None:
            nx.write_gml(G, save_random_path + '.gml')
        
        return G
    
    def make_subgraph(self, clustering_method='spectral'):
        """
        Create subgraphs using clustering method
        
        Args:
            clustering_method: 'spectral' for SpectralClustering or 'random' for random clustering
        """
        data, interactions, user_num, item_num = read_data_new(self.train_data_path)
        total_user_num = USER_NUM
        num_clients = NUM_CLIENTS
        
        G = nx.Graph()
        G.add_nodes_from(range(user_num), bipartite=0)
        G.add_nodes_from(range(total_user_num, total_user_num + item_num), bipartite=1)
        G.add_edges_from(interactions)
        
        adjacency_matrix = nx.adjacency_matrix(G)
        adjacency_array = sp.sparse.csr_matrix(adjacency_matrix)
        print("Dataset Name:", DATASET)
        print("Adjacency Array Shape:", adjacency_array.shape)
        
        if clustering_method == 'spectral':
            user_adjacency_matrix = adjacency_array[:user_num, :user_num]
            user_spectral = SpectralClustering(n_clusters=num_clients, affinity='precomputed', 
                                             n_init=100, assign_labels='discretize')
            user_labels = user_spectral.fit_predict(user_adjacency_matrix)
            
            item_adjacency_matrix = adjacency_array[user_num:, user_num:]
            item_spectral = SpectralClustering(n_clusters=num_clients, affinity='precomputed', 
                                             n_init=100, assign_labels='discretize')
            item_labels = item_spectral.fit_predict(item_adjacency_matrix)
        elif clustering_method == 'random':
            user_labels = [rd.randint(0, num_clients - 1) for _ in range(user_num)]
            item_labels = [rd.randint(0, num_clients - 1) for _ in range(item_num)]
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
        
        print("User Labels Shape:", len(user_labels))
        print("Item Labels Shape:", len(item_labels))
        
        user_labels_counter = Counter(user_labels)
        item_labels_counter = Counter(item_labels)
        print("User Labels Counter:", user_labels_counter)
        print("Item Labels Counter:", item_labels_counter)
        
        subgraphs = []
        total_edge_num = 0
        for i in range(num_clients):
            user_nodes = [node for node, label in zip(range(user_num), user_labels) if label == i]
            item_nodes = [node for node, label in zip(range(total_user_num, total_user_num + item_num), item_labels) if label == i]
            
            subgraph_nodes = user_nodes + item_nodes
            subgraph = G.subgraph(subgraph_nodes).copy()
            subgraph.add_edges_from(G.subgraph(subgraph_nodes).edges)
            subgraphs.append(subgraph)
            
            print("Subgraph", i + 1, "'s node number:", len(subgraph.nodes), 
                  ", edge number:", len(subgraph.edges))
            total_edge_num += len(subgraph.edges)
        
        print("Original Edge Number:", len(G.edges))
        print("Updated Edge Number:", total_edge_num)
        
        return subgraphs
    
    def compute_subgraph_degrees(self, subgraph):
        """
        Compute degree information for users and items in a subgraph
        
        Args:
            subgraph: NetworkX graph (subgraph)
            
        Returns:
            dict: Dictionary containing user_degrees and item_degrees
        """
        user_degrees = {}
        item_degrees = {}
        
        for node, node_data in subgraph.nodes(data=True):
            degree = subgraph.degree(node)
            if node_data.get('bipartite') == 0:
                user_degrees[int(node)] = int(degree)
            elif node_data.get('bipartite') == 1:
                item_degrees[int(node)] = int(degree)
        
        return {
            'user_degrees': user_degrees,
            'item_degrees': item_degrees
        }
    
    def save_subgraph(self, clustering_method='spectral', generate_random=True):
        """
        Save subgraphs to files
        
        Args:
            clustering_method: 'spectral' or 'random'
            generate_random: Whether to generate random graph
        """
        os.makedirs(self.clients_dir, exist_ok=True)
        
        random_graph = None
        if generate_random:
            random_graph = self.generate_random_graph()
        
        subgraphs = self.make_subgraph(clustering_method=clustering_method)
        
        for i in range(NUM_CLIENTS):
            subgraph = subgraphs[i]
            subgraph_path = self.subgraphs_path + str(i) + '.gml'
            nx.write_gml(subgraph, subgraph_path)
            
            # Compute and save degree information
            degree_info = self.compute_subgraph_degrees(subgraph)
            degree_path = self.clients_dir + 'degrees_' + str(i) + '.json'
            with open(degree_path, 'w') as f:
                json.dump(degree_info, f, indent=2)
            print(f"Saved degree information for subgraph {i+1}: "
                  f"users={len(degree_info['user_degrees'])}, "
                  f"items={len(degree_info['item_degrees'])}")
        
        if random_graph is not None:
            if RANDOMGRAPH_TYPE == 'random':
                save_random_path = self.clients_dir + self.RANDOM_GRAPH_PREFIX + '_random.gml'
            elif RANDOMGRAPH_TYPE == 'gnmk':
                save_random_path = self.clients_dir + self.RANDOM_GRAPH_PREFIX + '_gnmk.gml'
            nx.write_gml(random_graph, save_random_path)
            
            # Compute and save degree information for random graph
            degree_info = self.compute_subgraph_degrees(random_graph)
            if RANDOMGRAPH_TYPE == 'random':
                degree_path = self.clients_dir + 'degrees_random_random.json'
            elif RANDOMGRAPH_TYPE == 'gnmk':
                degree_path = self.clients_dir + 'degrees_random_gnmk.json'
            with open(degree_path, 'w') as f:
                json.dump(degree_info, f, indent=2)
            print(f"Saved degree information for random graph ({RANDOMGRAPH_TYPE}): "
                  f"users={len(degree_info['user_degrees'])}, "
                  f"items={len(degree_info['item_degrees'])}")
    
    def read_graph(self, read_random=True):
        """
        Read subgraphs from files
        
        Args:
            read_random: Whether to read random graph
        
        Returns:
            subgraphs: List of subgraph data
            random_graph: Random graph data (if read_random=True)
        """
        check_original = self.base_dir + "train_data.json"
        _, check, _, _ = read_data(check_original)
        origin_length = len(check)
        
        subgraphs = []
        random_graph = None
        total_user_num = USER_NUM
        total_item_num = ITEM_NUM
        total_edge_number = 0
        
        print("Total Node Number:", (total_user_num + total_item_num),
              ", Origin User Number:", total_user_num, ", Origin Item Number:", total_item_num)
        
        for i in range(NUM_CLIENTS):
            subgraph = nx.read_gml(self.subgraphs_path + str(i) + '.gml')
            user_nodes = [node for node, node_data in subgraph.nodes(data=True) 
                         if node_data.get('bipartite') == 0]
            item_nodes = [node for node, node_data in subgraph.nodes(data=True) 
                         if node_data.get('bipartite') == 1]
            
            interactions = []
            user_num = len(user_nodes)
            item_num = len(item_nodes)
            
            for user in user_nodes:
                for item in subgraph[user]:
                    interactions.append((user, item))
            rd.shuffle(interactions)
            subgraphs.append([subgraph, interactions, user_num, item_num])
            print("Subgraph", i + 1, "'s node number", len(subgraph.nodes), 
                  ", user number:", user_num, ", item number:", item_num, 
                  ", interaction number:", len(interactions))
            total_edge_number += len(interactions)
        
        if read_random:
            try:
                if RANDOMGRAPH_TYPE == 'random':
                    random_graph_path = self.clients_dir + self.RANDOM_GRAPH_PREFIX + '_random.gml'
                elif RANDOMGRAPH_TYPE == 'gnmk':
                    random_graph_path = self.clients_dir + self.RANDOM_GRAPH_PREFIX + '_gnmk.gml'
                else:
                    random_graph_path = self.clients_dir + self.RANDOM_GRAPH_PREFIX + '_random.gml'
                
                random_graph = nx.read_gml(random_graph_path)
                user_nodes = [node for node, node_data in random_graph.nodes(data=True) 
                             if node_data.get('bipartite') == 0]
                item_nodes = [node for node, node_data in random_graph.nodes(data=True) 
                             if node_data.get('bipartite') == 1]
                
                interactions = []
                user_num = len(user_nodes)
                item_num = len(item_nodes)
                
                for user in user_nodes:
                    for item in random_graph[user]:
                        interactions.append((user, item))
                rd.shuffle(interactions)
                random_graph = [random_graph, interactions, user_num, item_num]
                print("Random Graph's node number", len(random_graph[0].nodes), 
                      ", user number:", user_num, ", item number:", item_num, 
                      ", interaction number:", len(interactions))
            except FileNotFoundError:
                print("Random graph file not found, skipping...")
                random_graph = None
        
        print("Original Edge Number:", origin_length)
        print("Subgraph's Total Edge Number:", total_edge_number)
        
        if read_random:
            return subgraphs, random_graph
        else:
            return subgraphs
    
    def distribute_data_by_clients(self, clustering_method='spectral', val_ratio=0.1, seed=42):
        """
        Distribute train data to each client based on clustering results,
        then split each client's train data into train/validation (8:1 ratio)
        
        Args:
            clustering_method: 'spectral' or 'random' (must match the method used in make_subgraph)
            val_ratio: Validation ratio (default 0.1, so train:val = 8:1)
            seed: Random seed for reproducible splits
        """
        rd.seed(seed)
        
        # Get clustering labels from train data
        data, interactions, user_num, item_num = read_data_new(self.train_data_path)
        total_user_num = USER_NUM
        num_clients = NUM_CLIENTS
        
        G = nx.Graph()
        G.add_nodes_from(range(user_num), bipartite=0)
        G.add_nodes_from(range(total_user_num, total_user_num + item_num), bipartite=1)
        G.add_edges_from(interactions)
        
        adjacency_matrix = nx.adjacency_matrix(G)
        adjacency_array = sp.sparse.csr_matrix(adjacency_matrix)
        
        if clustering_method == 'spectral':
            user_adjacency_matrix = adjacency_array[:user_num, :user_num]
            user_spectral = SpectralClustering(n_clusters=num_clients, affinity='precomputed', 
                                             n_init=100, assign_labels='discretize')
            user_labels = user_spectral.fit_predict(user_adjacency_matrix)
            
            item_adjacency_matrix = adjacency_array[user_num:, user_num:]
            item_spectral = SpectralClustering(n_clusters=num_clients, affinity='precomputed', 
                                             n_init=100, assign_labels='discretize')
            item_labels = item_spectral.fit_predict(item_adjacency_matrix)
        elif clustering_method == 'random':
            user_labels = [rd.randint(0, num_clients - 1) for _ in range(user_num)]
            item_labels = [rd.randint(0, num_clients - 1) for _ in range(item_num)]
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
        
        # Load original train and test data (validation not needed, will be created from train)
        with open(self.train_data_path, 'r') as f:
            train_data = json.load(f)
        
        # Load test data if it exists, otherwise create empty list
        test_data = []
        if os.path.exists(self.test_data_path):
            with open(self.test_data_path, 'r') as f:
                test_data = json.load(f)
        else:
            print(f"Warning: Test data file not found at {self.test_data_path}, using empty test data")
        
        # Distribute data to each client
        os.makedirs(self.clients_dir, exist_ok=True)
        
        for client_idx in range(num_clients):
            # Get users assigned to this client
            client_users = [user for user, label in enumerate(user_labels) if label == client_idx]
            
            # Get items assigned to this client
            client_items = set()
            for user in client_users:
                if user < len(train_data):
                    client_items.update(train_data[user])
                if user < len(test_data):
                    client_items.update(test_data[user])
            
            # Filter train and test data for this client
            client_train_full = []  # Full train data before split
            client_test_data = []
            
            for user in client_users:
                if user < len(train_data):
                    # Only include items assigned to this client
                    user_train_items = [item for item in train_data[user] if item in client_items]
                    client_train_full.append(user_train_items)
                else:
                    client_train_full.append([])
                
                if user < len(test_data):
                    user_test_items = [item for item in test_data[user] if item in client_items]
                    client_test_data.append(user_test_items)
                else:
                    client_test_data.append([])
            
            # Split client's train data into train/validation (8:1 ratio)
            client_train_data = []
            client_val_data = []
            
            for user_train_items in client_train_full:
                num_items = len(user_train_items)
                if num_items < 2:
                    # If user has less than 2 items, put all in train
                    client_train_data.append(user_train_items)
                    client_val_data.append([])
                    continue
                
                # Calculate split sizes (8:1 ratio)
                num_val = max(1, int(num_items * val_ratio))
                num_train = num_items - num_val
                
                # Shuffle and split
                indices = list(range(num_items))
                rd.shuffle(indices)
                
                val_indices = sorted(indices[:num_val])
                train_indices = sorted(indices[num_val:])
                
                client_val_data.append([user_train_items[i] for i in val_indices])
                client_train_data.append([user_train_items[i] for i in train_indices])
            
            # Save client data
            with open(self.clients_dir + 'train_data' + str(client_idx) + '.json', 'w') as f:
                json.dump(client_train_data, f)
            with open(self.clients_dir + 'val_data' + str(client_idx) + '.json', 'w') as f:
                json.dump(client_val_data, f)
            with open(self.clients_dir + 'test_data' + str(client_idx) + '.json', 'w') as f:
                json.dump(client_test_data, f)
            
            # Count non-empty entries
            train_count = len([x for x in client_train_data if x])
            val_count = len([x for x in client_val_data if x])
            test_count = len([x for x in client_test_data if x])
            
            print(f"Client {client_idx}: Train={train_count}, Val={val_count}, Test={test_count}")


if __name__ == '__main__':
    make_subgraph = MakeSubgraph()
    
    make_subgraph.save_subgraph(clustering_method='spectral', generate_random=True)
    make_subgraph.read_graph(read_random=True)
    
    # Distribute train/val/test data to clients (maintains original split)
    make_subgraph.distribute_data_by_clients(clustering_method='spectral')