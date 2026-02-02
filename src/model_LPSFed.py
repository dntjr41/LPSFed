import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class LPSFed(nn.Module):
    def __init__(self, n_users, n_items, lr, lamda, emb_dim, layer, pre_train_latent_factor,
                 graph_embeddings, graph_conv, prediction, loss_function, generalization, optimization,
                 if_pretrain, top_k, if_transformation, activation, pooling, gpu_index, 
                 tau1, tau2, w_lambda, margin_ratio, graph_type=None):
        super(LPSFed, self).__init__()
        
        gpu_index = int(gpu_index)
        device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model_name = 'LPSFed'
        self.n_users = n_users
        self.n_items = n_items
        
        ## parameters
        self.lr = lr
        self.lamda = lamda
        self.emb_dim = emb_dim
        self.emb_dim_predict = (layer + 1) * emb_dim if pooling == 'Concat' else emb_dim
        if graph_conv == '1D': 
            if graph_embeddings is not None:
                if isinstance(graph_embeddings, np.ndarray):
                    self.frequency = graph_embeddings.shape[1]
                elif isinstance(graph_embeddings, torch.Tensor):
                    self.frequency = graph_embeddings.shape[1]
                else:
                    self.frequency = len(graph_embeddings[0]) if isinstance(graph_embeddings, (list, tuple)) else graph_embeddings.shape[1]
            else:
                raise ValueError("graph_embeddings cannot be None for 1D graph convolution")
        else: 
            if graph_embeddings is not None and isinstance(graph_embeddings, (list, tuple)) and len(graph_embeddings) == 2:
                if isinstance(graph_embeddings[0], np.ndarray):
                    self.frequency_U, self.frequency_V = graph_embeddings[0].shape[1], graph_embeddings[1].shape[1]
                elif isinstance(graph_embeddings[0], torch.Tensor):
                    self.frequency_U, self.frequency_V = graph_embeddings[0].shape[1], graph_embeddings[1].shape[1]
                else:
                    self.frequency_U, self.frequency_V = len(graph_embeddings[0]), len(graph_embeddings[1])
            else:
                raise ValueError("graph_embeddings must be a list/tuple of 2 elements for 2D graph convolution")
        self.layer = layer
        self.optimization = optimization
        
        self.tau1 = tau1
        self.tau2 = tau2
        self.w_lambda = w_lambda
        self.margin_ratio = margin_ratio
        self.avg_margin = nn.Parameter(torch.FloatTensor([0.0]))
        
        ## model parameters
        # Handle pre_train_latent_factor: can be list or tuple
        if isinstance(pre_train_latent_factor, (list, tuple)) and len(pre_train_latent_factor) == 2:
            self.U, self.V = pre_train_latent_factor
        else:
            # If not provided, use zeros
            self.U = torch.zeros(n_users, emb_dim, dtype=torch.float32)
            self.V = torch.zeros(n_items, emb_dim, dtype=torch.float32)
        if graph_conv == '1D':
            if graph_embeddings is not None:
                if isinstance(graph_embeddings, np.ndarray):
                    self.graph_emb = torch.FloatTensor(graph_embeddings).to(device)
                elif isinstance(graph_embeddings, torch.Tensor):
                    self.graph_emb = graph_embeddings.to(device)
                else:
                    self.graph_emb = torch.FloatTensor(np.array(graph_embeddings)).to(device)
            else:
                raise ValueError("graph_embeddings cannot be None for 1D graph convolution")
        
        else:
            if graph_embeddings is not None and isinstance(graph_embeddings, (list, tuple)) and len(graph_embeddings) == 2:
                self.graph_emb_U, self.graph_emb_V = graph_embeddings
                if isinstance(self.graph_emb_U, np.ndarray):
                    self.graph_emb_U = torch.FloatTensor(self.graph_emb_U).to(device)
                elif isinstance(self.graph_emb_U, torch.Tensor):
                    self.graph_emb_U = self.graph_emb_U.to(device)
                else:
                    self.graph_emb_U = torch.FloatTensor(np.array(self.graph_emb_U)).to(device)
                    
                if isinstance(self.graph_emb_V, np.ndarray):
                    self.graph_emb_V = torch.FloatTensor(self.graph_emb_V).to(device)
                elif isinstance(self.graph_emb_V, torch.Tensor):
                    self.graph_emb_V = self.graph_emb_V.to(device)
                else:
                    self.graph_emb_V = torch.FloatTensor(np.array(self.graph_emb_V)).to(device)
            else:
                raise ValueError("graph_embeddings must be a list/tuple of 2 elements for 2D graph convolution")
        
        ## network structure; model settings; and optimization setting
        self.graph_conv = graph_conv
        self.prediction = prediction
        self.loss_function = loss_function
        self.generalization = generalization.split('+')
        self.optimization = optimization
        self.if_pretrain = if_pretrain
        self.if_transformation = if_transformation
        self.activation = activation
        self.pooling = pooling
        self.graph_type = graph_type
        self.top_k = top_k
        
        ## Tensor definition
        self.users = torch.IntTensor([])
        self.pos_items = torch.IntTensor([])
        self.neg_items = torch.IntTensor([])
        self.keep_prob = torch.FloatTensor([])
        self.items_in_train_data = torch.FloatTensor([])
        
        ## learnable parameters
        if self.if_pretrain:
            self.user_embeddings = nn.Parameter(torch.Tensor(self.U))
            self.item_embeddings = nn.Parameter(torch.Tensor(self.V))
            self.user_pop_embeddings = nn.Parameter(torch.Tensor(self.U))
            self.item_pop_embeddings = nn.Parameter(torch.Tensor(self.V))
        else:
            self.user_embeddings = nn.Parameter(torch.randn(self.n_users, self.emb_dim, dtype=torch.float32) * 0.02 + 0.01)
            self.item_embeddings = nn.Parameter(torch.randn(self.n_items, self.emb_dim, dtype=torch.float32) * 0.02 + 0.01)
            self.user_pop_embeddings = nn.Parameter(torch.randn(self.n_users, self.emb_dim, dtype=torch.float32) * 0.02 + 0.01)
            self.item_pop_embeddings = nn.Parameter(torch.randn(self.n_items, self.emb_dim, dtype=torch.float32) * 0.02 + 0.01)
        
        if graph_conv == '1D':
            self.kernel = nn.ParameterList([nn.Parameter(torch.randn(self.frequency, dtype=torch.float32) * 0.02 + 0.01) for _ in range(layer)])
        else:
            self.kernel_U = nn.ParameterList([nn.Parameter(torch.randn(self.frequency_U, dtype=torch.float32) * 0.02 + 0.01) for _ in range(layer)])
            self.kernel_V = nn.ParameterList([nn.Parameter(torch.randn(self.frequency_V, dtype=torch.float32) * 0.02 + 0.01) for _ in range(layer)])

        if self.if_transformation: 
            self.transformation = nn.ParameterList([nn.Parameter(torch.randn(self.emb_dim, self.emb_dim, dtype=torch.float32) * 0.02 + 0.01) for l in range(self.layer)])
        
        if self.pooling == 'Sum': 
            self.layer_weight = [1 / (l + 1) ** 1 for l in range(self.layer + 1)]
        if self.pooling[0: 3] == 'MLP':
            self.pooling_mlp_layer = 1 # int(self.pooling[3:])
            self.pooling_layer_size = [(self.layer + 1) * self.emb_dim] + [(self.pooling_mlp_layer - l) * self.emb_dim_predict for l in range(self.pooling_mlp_layer)]
            self.pooling_W = nn.ParameterList([nn.Parameter(torch.randn(self.pooling_layer_size[l], self.pooling_layer_size[l + 1]) * 0.01) for l in range(self.pooling_mlp_layer)])
            self.pooling_b = nn.ParameterList([nn.Parameter(torch.randn(self.pooling_layer_size[l + 1]) * 0.01) for l in range(self.pooling_mlp_layer)])
        if self.prediction[0: 3] == 'MLP':
            self.prediction_mlp_layer = 1 # int(self.prediction[3:])
            self.prediction_layer_size = [3 * self.emb_dim_predict] + [self.emb_dim_predict] * (self.prediction_mlp_layer - 1) + [1]
            self.prediction_W = nn.ParameterList([nn.Parameter(torch.randn(self.prediction_layer_size[l], self.prediction_layer_size[l + 1]) * 0.01) for l in range(self.prediction_mlp_layer)])
            self.prediction_b = nn.ParameterList([nn.Parameter(torch.randn(self.prediction_layer_size[l + 1]) * 0.01) for l in range(self.prediction_mlp_layer)])
        
        ## update parameters
        self.var_list = [self.user_embeddings, self.item_embeddings]  ## learnable parameter list
        # Add popularity embeddings to learnable parameters for BC-Loss
        if self.loss_function == 'BC':
            self.var_list += [self.user_pop_embeddings, self.item_pop_embeddings]
        if self.graph_conv == '1D': self.var_list += self.kernel
        else: 
            self.var_list += self.kernel_U
            self.var_list += self.kernel_V
        if self.if_transformation: self.var_list += self.transformation
        if self.pooling[0: 3] == 'MLP': 
            self.var_list += self.pooling_W
            self.var_list += self.pooling_b
        if self.prediction[0: 3] == 'MLP': 
            self.var_list += self.prediction_W
            self.var_list += self.prediction_b
        
        if optimization == 'SGD': self.optimizer = optim.SGD(self.var_list, lr=self.lr, weight_decay=self.lamda)
        elif optimization == 'RMSProp': self.optimizer = optim.RMSprop(self.var_list, lr=self.lr, weight_decay=self.lamda)
        elif optimization == 'Adam': self.optimizer = optim.Adam(self.var_list, lr=self.lr, weight_decay=self.lamda)
        elif optimization == 'Adagrad': self.optimizer = optim.Adagrad(self.var_list, lr=self.lr, weight_decay=self.lamda)
        
    # Convolutional layer
    def convolutional_layers(self, u_embeddings, i_embeddings):
        embeddings = torch.cat([u_embeddings, i_embeddings], dim=0)
        
        if self.graph_conv == '1D':
            graph_emb_size = self.graph_emb.size(0)
            embeddings_size = embeddings.size(0)
            
            if graph_emb_size != embeddings_size:
                if embeddings_size > graph_emb_size:
                    embeddings = embeddings[:graph_emb_size, :]
                else:
                    padding_size = graph_emb_size - embeddings_size
                    embeddings = torch.cat([embeddings, torch.zeros(padding_size, embeddings.size(1))], dim=0)
        
        if self.pooling in ['Sum', 'Product']: self.all_embeddings = embeddings
        else: self.all_embeddings = [embeddings]
        
        for l in range(self.layer):
            if self.graph_conv == '1D':
                graph_emb_size = self.graph_emb.size(0)
                embeddings_size = embeddings.size(0)
                
                if graph_emb_size != embeddings_size:
                    if embeddings_size > graph_emb_size:
                        embeddings_adjusted = embeddings[:graph_emb_size, :]
                    else:
                        padding_size = graph_emb_size - embeddings_size
                        embeddings_adjusted = torch.cat([embeddings, torch.zeros(padding_size, embeddings.size(1))], dim=0)
                    graph_emb_adjusted = self.graph_emb
                else:
                    graph_emb_adjusted = self.graph_emb
                    embeddings_adjusted = embeddings
                
                embeddings = torch.mm(torch.mm(graph_emb_adjusted.to(self.device), torch.diag(self.kernel[l].to(self.device))),
                                      torch.mm(graph_emb_adjusted.T.to(self.device), embeddings_adjusted))
            else:
                embeddings_U, embeddings_V = torch.split(embeddings, [self.n_users, self.n_items], dim=1)
                embeddings_U = torch.mm(torch.mm(self.graph_emb_U, torch.diag(self.kernel_U[l])),
                                        torch.mm(self.graph_emb_U.T, embeddings_U))
                embeddings_V = torch.mm(torch.mm(self.graph_emb_V, torch.diag(self.kernel_V[l])),
                                        torch.mm(self.graph_emb_V.T, embeddings_V))
                embeddings = torch.cat([embeddings_U, embeddings_V], dim=1)
            
            if self.if_transformation: embeddings = torch.mm(embeddings, self.transformation[l])
            
            if self.activation == 'Sigmoid': embeddings = torch.sigmoid(embeddings)
            elif self.activation == 'Tanh': embeddings = torch.tanh(embeddings)
            elif self.activation == 'ReLU': embeddings = F.relu(embeddings)
            elif self.activation == 'LeakyReLU': embeddings = F.leaky_relu(embeddings)
            
            if self.pooling == 'Sum': 
                if hasattr(self, 'all_embeddings') and isinstance(self.all_embeddings, torch.Tensor):
                    if self.all_embeddings.size(0) != embeddings.size(0):
                        if self.all_embeddings.size(0) > embeddings.size(0):
                            self.all_embeddings = self.all_embeddings[:embeddings.size(0), :]
                        else:
                            padding_size = embeddings.size(0) - self.all_embeddings.size(0)
                            self.all_embeddings = torch.cat([self.all_embeddings, torch.zeros(padding_size, self.all_embeddings.size(1))], dim=0)
                self.all_embeddings += embeddings * self.layer_weight[l + 1]
            elif self.pooling == 'Product': self.all_embeddings *= torch.sigmoid(embeddings)
            else: self.all_embeddings += [embeddings]
    
    # Pooling Layer
    def pooling_layer(self):
        if self.pooling == 'Concat':
            self.all_embeddings = torch.cat(self.all_embeddings, dim=1)
        elif self.pooling == 'Max':
            self.all_embeddings = torch.stack(self.all_embeddings, dim=0)
            self.all_embeddings, _ = torch.max(self.all_embeddings, dim=0)
        elif self.pooling[0:3] == 'MLP':
            self.all_embeddings = torch.cat(self.all_embeddings, dim=1)
            self.all_embeddings = torch.tanh(self.MLP(self.all_embeddings, self.pooling_W, self.pooling_b))
        
        if 'L2Norm' in self.generalization:
            self.all_embeddings = torch.nn.functional.normalize(self.all_embeddings, p=2, dim=1)
        
        actual_size = self.all_embeddings.size(0)
        expected_size = self.n_users + self.n_items
        
        if actual_size != expected_size:
            user_ratio = self.n_users / expected_size
            item_ratio = self.n_items / expected_size
            
            user_size = int(actual_size * user_ratio)
            item_size = actual_size - user_size
            
            self.user_all_embeddings, self.item_all_embeddings = torch.split(self.all_embeddings, [user_size, item_size], dim=0)
        else:
            self.user_all_embeddings, self.item_all_embeddings = torch.split(self.all_embeddings, [self.n_users, self.n_items], dim=0) # type: ignore
    
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop, keep_porb):
        if not isinstance(users, torch.Tensor):
            users = torch.tensor(users, device=self.device)
        if pos_items is not None and not isinstance(pos_items, torch.Tensor):
            pos_items = torch.tensor(pos_items, device=self.device)
        if neg_items is not None and not isinstance(neg_items, torch.Tensor):
            neg_items = torch.tensor(neg_items, device=self.device)
        
        self.convolutional_layers(self.user_embeddings, self.item_embeddings)
        self.pooling_layer()
        
        if pos_items is None and neg_items is None:
            user_size = self.user_all_embeddings.size(0)
            item_size = self.item_all_embeddings.size(0)
            
            if not isinstance(users, torch.Tensor):
                users = torch.tensor(users, device=self.device)
            users_adjusted = torch.clamp(users, 0, user_size - 1)
            user_embeddings = self.user_all_embeddings[users_adjusted]
            
            if self.prediction[0: 3] == 'MLP': 
                ratings = self.get_all_ratings(user_embeddings, self.item_all_embeddings, self.prediction_W, self.prediction_b)
            else:
                ratings = torch.mm(user_embeddings, self.item_all_embeddings.T)
            
            # Ensure k doesn't exceed item_size
            k = min(self.top_k, item_size)
            top_items = torch.topk(ratings, k=k, dim=1, largest=True, sorted=True)
            test_top_items = top_items.indices
            
            return user_embeddings, ratings, top_items, test_top_items
        
        user_size = self.user_all_embeddings.size(0)
        item_size = self.item_all_embeddings.size(0)
        
        users_adjusted = torch.clamp(users, 0, user_size - 1)
        pos_items_adjusted = torch.clamp(pos_items, 0, item_size - 1)
        neg_items_adjusted = torch.clamp(neg_items, 0, item_size - 1)
        
        pop_user_size = self.user_pop_embeddings.size(0)
        pop_item_size = self.item_pop_embeddings.size(0)
        
        pop_users_adjusted = torch.clamp(users, 0, pop_user_size - 1)
        pop_pos_items_adjusted = torch.clamp(pos_items, 0, pop_item_size - 1)
        pop_neg_items_adjusted = torch.clamp(neg_items, 0, pop_item_size - 1)
        
        u_pop_embeddings = self.user_pop_embeddings[pop_users_adjusted]
        pos_i_pop_embeddings = self.item_pop_embeddings[pop_pos_items_adjusted]
        neg_i_pop_embeddings = self.item_pop_embeddings[pop_neg_items_adjusted]
        
        u_pop_embeddings = F.normalize(u_pop_embeddings, dim=-1)
        pos_i_pop_embeddings = F.normalize(pos_i_pop_embeddings, dim=-1)
        neg_i_pop_embeddings = F.normalize(neg_i_pop_embeddings, dim=-1)
        
        # Popularity embeddings similarity (for popularity-aware contrastive loss)
        pos_ratings_pop = torch.sum(u_pop_embeddings * pos_i_pop_embeddings, dim=-1)
        neg_ratings_pop = torch.matmul(torch.unsqueeze(u_pop_embeddings, 1),
                                       neg_i_pop_embeddings.unsqueeze(2)).squeeze(dim=1)
        ratings_pop = torch.cat([pos_ratings_pop[:, None], neg_ratings_pop], dim=1)
        
        # Compute bias-aware margin from popularity embeddings
        # Higher popularity similarity -> smaller margin needed (less bias correction)
        pos_ratings_margin = pos_ratings_pop
        avg_ratings_margin_scalar = torch.mean(pos_ratings_margin)
        # Margin ratio controls how much to use personalized vs averaged margin
        updated_pos_ratings_margin = pos_ratings_margin * (1 - self.margin_ratio) + avg_ratings_margin_scalar * self.margin_ratio
        
        # Popularity contrastive loss components (InfoNCE style with tau2)
        pop_numerator = torch.exp(pos_ratings_pop / self.tau2)
        pop_denominator = torch.sum(torch.exp(ratings_pop / self.tau2), dim=-1)
        
        u_embeddings = self.user_all_embeddings[users_adjusted]
        pos_i_embeddings = self.item_all_embeddings[pos_items_adjusted]
        neg_i_embeddings = self.item_all_embeddings[neg_items_adjusted]
        
        u_embeddings = F.normalize(u_embeddings, dim=-1)
        pos_i_embeddings = F.normalize(pos_i_embeddings, dim=-1)
        neg_i_embeddings = F.normalize(neg_i_embeddings, dim=-1)
        
        # Main embeddings similarity (for main contrastive loss)
        pos_ratings_main = torch.sum(u_embeddings * pos_i_embeddings, dim=-1)
        neg_ratings_main = torch.matmul(torch.unsqueeze(u_embeddings, 1),
                                       neg_i_embeddings.unsqueeze(2)).squeeze(dim=1)
        
        # Apply bias-aware margin to main ratings
        # Margin is based on popularity: higher popularity -> smaller margin needed
        # updated_pos_ratings_margin is already computed from popularity embeddings
        # Convert margin to a scaling factor for the main ratings
        margin_factor = torch.sigmoid(updated_pos_ratings_margin)
        # Adjust main ratings with bias-aware margin
        pos_ratings = pos_ratings_main * (1 + margin_factor * self.margin_ratio)
        
        # For negative ratings, we don't apply margin (or apply negative margin)
        neg_ratings = neg_ratings_main
        
        # Prepare ratings for contrastive loss
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)
        
        # Main contrastive loss components (InfoNCE style)
        main_numerator = torch.exp(pos_ratings / self.tau1)
        main_denominator = torch.sum(torch.exp(ratings / self.tau1), dim=1)

        avg_embedding = 0
        if self.graph_type == 'random':
            all_embedding = torch.cat([self.user_all_embeddings, self.item_all_embeddings], dim=0)
            avg_embedding = torch.mean(all_embedding, dim=1)
            
        self.avg_margin.data.fill_(avg_ratings_margin_scalar.item())
        
        return pos_ratings, neg_ratings, [], pos_i_embeddings, neg_i_embeddings, pop_numerator, pop_denominator, main_numerator, main_denominator
    
    def freeze_pop(self):
        self.user_pop_embeddings.requires_grad_(False)
        self.item_pop_embeddings.requires_grad_(False)
        
    def MLP(self, x, W, b):
        x = x
        W = [W[l] for l in range(len(W))]
        b = [b[l] for l in range(len(b))]
        
        for l in range(len(W) - 1):
            x = torch.tanh(x @ W[l] + b[l])
            if 'DropOut' in self.generalization:
                x = torch.F.dropout(x, self.keep_prob)
        return x @ W[-1] + b[-1]

    def get_all_ratings(self, user_emb, item_emb, W, b):
        user_num = user_emb.shape[0]
        item_num = item_emb.shape[0]
        user_emb_extend = user_emb.repeat(1, item_num).view(-1, self.emb_dim_predict)
        item_emb_extend = item_emb.repeat(user_num, 1)
        score = self.MLP(torch.cat([user_emb_extend, item_emb_extend, user_emb_extend * item_emb_extend], dim=1), W, b)
        score = score.view(user_num, -1)
        score = torch.arccos(torch.tanh(score))
        return score

    def regularization(self, reg_list):
        reg = 0
        for para in reg_list:
            reg += para.norm(2)
        return reg
    
    def bc_loss(self, pop_numerator, pop_denominator, numerator, denominator):
        """
        BC-Loss: Bias-aware Contrastive Loss
        Combines popularity loss and main loss with learnable weights
        
        Args:
            pop_numerator: exp(pos_pop_rating / tau2) for popularity embeddings
            pop_denominator: sum of exp(ratings / tau2) for popularity embeddings
            numerator: exp(pos_main_rating / tau1) for main embeddings
            denominator: sum of exp(ratings / tau1) for main embeddings
        """
        # Popularity loss: -log(exp(pos_pop/tau2) / sum(exp(ratings/tau2)))
        # This encourages learning popularity bias
        epsilon = 1e-10
        pop_loss = -torch.mean(torch.log(pop_numerator / (pop_denominator + epsilon) + epsilon))
        
        # Main loss: -log(exp(pos_main/tau1) / sum(exp(ratings/tau1)))
        # This encourages learning actual user preferences
        main_loss = -torch.mean(torch.log(numerator / (denominator + epsilon) + epsilon))
        
        # Combined loss: weighted sum of popularity and main loss
        # w_lambda controls the trade-off between learning bias vs preference
        loss = self.w_lambda * pop_loss + (1 - self.w_lambda) * main_loss
        
        return loss

    def bpr_loss(self, pos_scores, neg_scores):
        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores))
        loss = -torch.sum(maxi)
        return loss
    
    def cross_entropy_loss(self, pos_scores, neg_scores):
        """
        Cross Entropy Loss for recommendation
        Treats recommendation as multi-class classification
        """
        # Combine positive and negative scores
        scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        # Create labels: 0 for positive item (first position)
        labels = torch.zeros(pos_scores.size(0), dtype=torch.long, device=pos_scores.device)
        loss = F.cross_entropy(scores, labels)
        return loss
    
    def mse_loss(self, pos_scores, neg_scores):
        """
        MSE Loss: minimize difference between positive and negative scores
        """
        # Positive scores should be higher than negative scores
        # Target: positive score = 1, negative scores = 0
        pos_target = torch.ones_like(pos_scores)
        neg_target = torch.zeros_like(neg_scores)
        
        pos_loss = F.mse_loss(pos_scores, pos_target)
        neg_loss = F.mse_loss(neg_scores, neg_target)
        loss = pos_loss + neg_loss
        return loss