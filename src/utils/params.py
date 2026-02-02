model = 1           # 0:MF, 1:LGCN
dataset = 2         # 0:Amazon, 1:Gowalla, 2:ML-1M, 3:Yelp, 4:Tmall
pred_dim = 64       # predictive embedding dimensionality 
seed = 42

## parameters about experiment setting
GPU_INDEX = 0
DATASET = ['Amazon', 'Gowalla', 'ML-1M', 'Yelp2018', 'Tmall_buy'][dataset]
# USER_NUM = [52643, 29858, 6022, 31668][dataset]
# ITEM_NUM = [91599, 40981, 3706, 36602][dataset]
USER_NUM = [52643, 24070, 5856, 31668, 151194][dataset]
ITEM_NUM = [61351, 28846, 2995, 27205, 42501][dataset]

RAND_USER_NUM = [13161, 6437, 1506, 7917, 48423][dataset]

ORIGIN_USER_NUM = [52643, 29858, 6022, 31668, 885759][dataset]
ORIGIN_ITEM_NUM = [91599, 40981, 3706, 36602, 1144124][dataset]

MODEL_list = ['MF', 'LGCN']
MODEL = MODEL_list[model]

## hyperparameters of all models
LR_list = [[0.05, 0.001], [0.05, 0.0005], [0.05, 0.0005], [0.05, 0.001], [0.05, 0.001]]
LAMDA_list = [[0.02, 0.02], [0.02, 0.02], [0.02, 0.02], [0.02, 0.02], [0.02, 0.02]]
LAYER_list = [[0, 2], [0, 2], [0, 2], [0, 2], [0, 2]]
LR = LR_list[dataset][model]
LAMDA = LAMDA_list[dataset][model]
LAYER = LAYER_list[dataset][model]

# dimensionality of the embedding layer
EMB_list = [pred_dim, pred_dim]
EMB_DIM = EMB_list[model]
BATCH_SIZE = 4096
TEST_USER_BATCH_list = [32, 128, 32, 64, 512]
TEST_USER_BATCH = TEST_USER_BATCH_list[dataset]
N_EPOCH = 200
IF_PRETRAIN = [False, True][1]
TEST_VALIDATION = 'Validation'  # can be changed automatically
TOP_K = [5, 10, 20][2]

## hyperparameters for LCFN and LGCN
FREQUENCY_USER_list = [100, 100, 100, 100, 100]
FREQUENCY_ITEM_list = [50, 50, 50, 50, 50]
FREQUENCY_USER = FREQUENCY_USER_list[dataset]
FREQUENCY_ITEM = FREQUENCY_ITEM_list[dataset]

## hyperparameters for LGCN
FREQUENCY = 64
KEEP_PORB = 0.9
SAMPLE_RATE = 1
GRAPH_CONV = ['1D', '2D_graph', '2D_hyper_graph'][0]
PREDICTION = ['InnerProduct', 'MLP3'][0]
LOSS_FUNCTION = ['BPR', 'CrossEntropy', 'MSE', 'BC'][3]
GENERALIZATION = ['Regularization', 'DropOut', 'Regularization+DropOut', 'L2Norm'][0]
OPTIMIZATION = ['SGD', 'Adagrad', 'RMSProp', 'Adam'][2]
IF_TRASFORMATION = [False, True][0]                           # 0 for not having transformation matrix,1 for having
ACTIVATION = ['None', 'Tanh', 'Sigmoid', 'ReLU'][0]          # select the activation function
POOLING = ['Concat', 'Sum', 'Max', 'Product', 'MLP3'][1]    # select the pooling strategy, the layer of mlp is also changable
if POOLING == 'Concat': EMB_DIM = int(pred_dim/(LAYER+1))

## parameters about model setting (selective for model LGCN)
PROP_DIM = 64
PROP_EMB = ['RM', 'SF', 'PE'][1]
IF_NORM = [False, True][0]

## For federated learning parameters
NUM_CLIENTS = 4
RANDOMGRAPH_TYPE = ['random', 'gnmk'][1]

all_para = [GPU_INDEX, DATASET, MODEL, LR, LAMDA, LAYER, EMB_DIM, BATCH_SIZE, TEST_USER_BATCH, N_EPOCH, IF_PRETRAIN,
            TEST_VALIDATION, TOP_K, FREQUENCY_USER, FREQUENCY_ITEM, FREQUENCY, KEEP_PORB, SAMPLE_RATE, GRAPH_CONV,
            PREDICTION, LOSS_FUNCTION, GENERALIZATION, OPTIMIZATION, IF_TRASFORMATION, ACTIVATION, POOLING, PROP_DIM,
            PROP_EMB, IF_NORM, NUM_CLIENTS, USER_NUM]

# If the client is updated every 10 epochs for the global model

GLOBAL_UPDATE_EPOCH = 15
COLD_THRESHOLD = 10

# FED_RATIO = [['rg', 'avg'], ['rg', 'per'], ['avg', 'avg'], ['avg', 'per']][0]
# 'rg' for random graph eigenvalues, 'avg' for average eigenvalues
# 'avg' for average, 'per' for personalized
FED_RATIO = [['rg', 'avg'], ['rg', 'per'], ['avg', 'avg'], ['avg', 'per']][1]
FED_METHOD = ['split', 'fedavg', 'lpsfed'][2]

# BC-Loss parameters
tau1 = 0.07 # Temperature for the L1
tau2 = 0.08  # Temperature for the L_BC
w_lambda = 0 # Strength of the Bias-margin
freeze_epoch = 10 # Freeze the model for the first 5 epochs
margin_ratio = 0.5  # Margin Ratio for the BC-Loss (Averaging, Personalized)

add_para = [seed, RANDOMGRAPH_TYPE, FED_RATIO, FED_METHOD, GLOBAL_UPDATE_EPOCH, COLD_THRESHOLD, tau1, tau2, w_lambda, freeze_epoch, margin_ratio]