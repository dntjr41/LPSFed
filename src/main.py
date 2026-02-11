from utils.params import all_para, add_para
from utils.read_subgraph_data import read_all_data, set_seed
import os

from test import test
from train_model import train_model
import time

if __name__ == "__main__":
    start_time = time.time()
    ## setting tuning strategies here
    path_excel_dir = 'experiment_result/' + all_para[1] + '_' + all_para[2] + '_'
    tuning_method = ['tuning', 'fine_tuning', 'cross_tuning', 'coarse_tuning', 'test'][4]  ## set here to tune model or test model
    
    ## initial hyperparameter settings
    lr_coarse, lamda_coarse = 0.001, 0.01
    lr_fine, lamda_fine = 0.0005, 0.1
    
    ## repeat numbers
    min_num_coarse, max_num_coarse = 3, 5
    min_num_fine, max_num_fine = 10, 50
    iter_num_test = 20
    
    ## select hyperparameters for different model
    para = all_para[0: 13]
    if all_para[2] == 'LGCN': 
        para += all_para[13: 26]
        para += all_para[29: 31]
    para_name = ['GPU_INDEX', 'DATASET', 'MODEL', 'LR', 'LAMDA', 'LAYER', 'EMB_DIM', 'BATCH_SIZE', 'TEST_USER_BATCH', 'N_EPOCH', 'IF_PRETRAIN', 'TEST_VALIDATION', 'TOP_K']
    if all_para[2] == 'LGCN':
        para_name += ['FREQUENCY_USER', 'FREQUENCY_ITEM', 'FREQUENCY', 'KEEP_PORB', 'SAMPLE_RATE', 'GRAPH_CONV', 'PREDICTION', 'LOSS_FUNCTION', 
                      'GENERALIZATION', 'OPTIMIZATION', 'IF_TRASFORMATION', 'ACTIVATION', 'POOLING', 'NUM_CLIENTS', 'USER_NUM']
    
    # if testing the model, we need to read in test set
    if tuning_method == 'test': all_para[11] = para[11] = 'Test'
    
    # Include gamma and omega if available (for backward compatibility)
    if len(add_para) >= 13:
        add_para = add_para[0:13]
        add_para_name = ['SEED', 'RANDOMGRAPH_TYPE', 'FED_RATIO', 'FED_METHOD', 'GLOBAL_UPDATE_EPOCH', 'COLD_THRESHOLD', 'tau1', 'tau2', 'w_lambda', 'freeze_epoch', 'margin_ratio', 'gamma', 'omega']
    else:
        add_para = add_para[0:11]
        add_para_name = ['SEED', 'RANDOMGRAPH_TYPE', 'FED_RATIO', 'FED_METHOD', 'GLOBAL_UPDATE_EPOCH', 'COLD_THRESHOLD', 'tau1', 'tau2', 'w_lambda', 'freeze_epoch', 'margin_ratio']

    ## read data
    data = read_all_data(all_para, add_para[1])
    # IF_PRETRAIN is now at index 7 (data[7]) or data[-4] after adding val_datas
    para[10] = data[7]  # IF_PRETRAIN
    set_seed(add_para[0])
    
    # all_para[0] = str(all_para[0])
    # os.environ["CUDA_VISIBLE_DEVICES"] = all_para[0]
    
    end_time = time.time()
    print("Data reading time: ", end_time - start_time)
    if tuning_method == 'test': test(path_excel_dir, para_name, para, add_para_name, add_para, data, iter_num_test)