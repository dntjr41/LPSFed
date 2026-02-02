import numpy as np

# Mapping The Training, Test Data
# Index of user and item in each client is different
    
def dataset_preprocessing(num_clients, train_datas, test_datas):
    
    sorted_interactions = []
    for i in range(num_clients):
        sorted_interactions.append(sorted(train_datas[i][1], key=lambda x: x[0]))
    
    removed_test_datas = []
    
    for i in range(num_clients):
        unique_users = set(user[0] for user in sorted_interactions[i])
        mapped_dict_user = {val: idx for idx, val in enumerate(unique_users)}
        transformed_user = [(mapped_dict_user[item[0]], item[1]) for item in sorted_interactions[i]]
        
        mapped_dict_item = {idx: val for idx, val in enumerate(train_datas[i][4])}
        fliped_dict_item = {val: idx for idx, val in enumerate(train_datas[i][4])}
        # Filter out items that don't exist in train data's item_nodes
        transformed_item = [(item[0], fliped_dict_item[item[1]]) for item in transformed_user if item[1] in fliped_dict_item]
        train_datas[i][1] = transformed_item
        
        for sublist in train_datas[i][0]:
            for idx,val in enumerate(sublist):
                if val in mapped_dict_item.values():
                    key = list(mapped_dict_item.keys())[list(mapped_dict_item.values()).index(val)]
                    sublist[idx] = key
        
        # Process test/validation data: map items and filter out items not in train data
        for sublist in test_datas[i]:
            if not sublist:  # Skip empty lists
                continue
            for idx,val in enumerate(sublist):
                if val in mapped_dict_item.values():
                    key = list(mapped_dict_item.keys())[list(mapped_dict_item.values()).index(val)]
                    sublist[idx] = key
                # If item not in train data, it will be filtered out later in filtered_values
        
        # Filter out items that don't exist in train data's item_nodes
        # Also handle empty validation/test data
        filtered_values = []
        for sublist in test_datas[i]:
            if not sublist:  # Handle empty lists (e.g., empty validation data)
                filtered_values.append([])
            else:
                filtered_sublist = [val for val in sublist if val in train_datas[i][4]]
                filtered_values.append(filtered_sublist)
        removed_test_datas.append(filtered_values)

    return removed_test_datas

def testset_preprocessing(num_clients, train_datas, test_datas, TOP_K, TEST_USER_BATCH, KEEP_PORB):
    para_test = []
    for i in range(num_clients):
        test = [train_datas[i][0], test_datas[i], train_datas[i][2], train_datas[i][3], TOP_K, TEST_USER_BATCH, KEEP_PORB]
        para_test.append(test)
    
    return para_test

def get_item_degrees(interaction_data):
    item_degrees = {}
    
    # Iterate over each interaction (user_id, item_id)
    for interaction in interaction_data:
        item_id = interaction[1]  # Extracting item_id from interaction
        
        # If item_id already exists in item_degrees, increment its degree
        if item_id in item_degrees:
            item_degrees[item_id] += 1
        # If item_id is encountered for the first time, initialize its degree to 1
        else:
            item_degrees[item_id] = 1
    
    return item_degrees

def get_all_degrees(interaction_data):
    user_degrees = {}
    item_degrees = {}
    
    # Iterate over each interaction (user_id, item_id)
    for interaction in interaction_data:
        user_id = interaction[0]  # Extracting user_id from interaction
        item_id = interaction[1]  # Extracting item_id from interaction
        
        # If user_id already exists in user_degrees, increment its degree
        if user_id in user_degrees:
            user_degrees[user_id] += 1
        # If user_id is encountered for the first time, initialize its degree to 1
        else:
            user_degrees[user_id] = 1
        
        # If item_id already exists in item_degrees, increment its degree
        if item_id in item_degrees:
            item_degrees[item_id] += 1
        # If item_id is encountered for the first time, initialize its degree to 1
        else:
            item_degrees[item_id] = 1
            
    return user_degrees, item_degrees

def get_pop_index(degree):
    if degree <= 10: return 0
    elif degree <= 20: return 1
    elif degree <= 30: return 2
    elif degree <= 40: return 3
    elif degree <= 50: return 4
    elif degree <= 100: return 5
    elif degree <= 200: return 6
    elif degree <= 300: return 7
    elif degree <= 400: return 8
    elif degree <= 500: return 9
    else: return 10