import random
import json
from sklearn.model_selection import train_test_split
import pandas as pd

# Sort the dataset
def sort_txt(path):
    data_json = []
    with open(path+'train.txt', 'r') as file:
        lines = file.readlines()

    data = [list(map(int, line.split())) for line in lines]
    sorted_data = sorted(data, key=lambda x: x[0])

    with open(path+'sorted_train.txt', 'w') as file:
        for line in sorted_data:
            file.write(' '.join(map(str, line)) + '\n')
            
    with open(path+'test.txt', 'r') as file:
        lines = file.readlines()

    data = [list(map(int, line.split())) for line in lines]
    sorted_data = sorted(data, key=lambda x: x[0])

    with open(path+'sorted_test.txt', 'w') as file:
        for line in sorted_data:
            file.write(' '.join(map(str, line)) + '\n')
###########################################################################

def combining_train_test(path):
    data1 = {}
    with open(path+'train.txt', 'r') as file1:
        for line in file1:
            user_id, *items = line.split()
            data1.setdefault(user_id, []).extend(items)

    data2 = {}
    with open(path+'test.txt', 'r') as file2:
        for line in file2:
            user_id, *items = line.split()
            data2.setdefault(user_id, []).extend(items)

    combined_data = {}
    for user_id, items in data1.items():
        if user_id in data2:
            combined_data[user_id] = items + data2[user_id]
        else:
            combined_data[user_id] = items

    with open(path+'all_dataset.txt', 'w') as output_file:
        for user_id, items in combined_data.items():
            output_file.write(f"{user_id} {' '.join(items)}\n")

###########################################################################
# txt to json
def txt_to_json(path):
    sort_txt(path)
    
    train_json =[]
    with open(path+'sorted_train.txt', 'r') as file:
        for line in file:
            items = [int(item) for item in line.strip().split()]
            item_ids = items[1:]
            train_json.append(item_ids)

    with open(path+'train.json', 'w') as file:
        json.dump(train_json, file)
    
    test_json =[]
    with open(path+'sorted_test.txt', 'r') as file:
        for line in file:
            items = [int(item) for item in line.strip().split()]
            item_ids = items[1:]
            test_json.append(item_ids)

    with open(path+'test.json', 'w') as file:
        json.dump(test_json, file)
        
###########################################################################
# Split train, test (9:1 ratio, no validation)
def split_data_by_user(path, train_ratio=0.9, test_ratio=0.1, seed=42):
    """
    Split data into train and test sets with 9:1 ratio.
    Validation set is excluded.
    """
    random.seed(seed)
    
    data_json = []
    with open(path+'train.json', 'r') as file:
        data_json = json.load(file)
    
    train_data = []
    test_data = []
    
    for user_items in data_json:
        num_items = len(user_items)
        if num_items < 2:
            # If user has less than 2 items, put all in train
            train_data.append(user_items)
            test_data.append([])
            continue
        
        # Calculate split sizes (9:1 ratio)
        num_test = max(1, int(num_items * test_ratio))
        num_train = num_items - num_test
        
        # Shuffle and split
        indices = list(range(num_items))
        random.shuffle(indices)
        
        test_indices = sorted(indices[:num_test])
        train_indices = sorted(indices[num_test:])
        
        test_data.append([user_items[i] for i in test_indices])
        train_data.append([user_items[i] for i in train_indices])
    
    # Save train and test data
    with open(path+'train_data.json', 'w') as train_file:
        json.dump(train_data, train_file)
    
    with open(path+'test_data.json', 'w') as test_file:
        json.dump(test_data, test_file)
    
    # Count non-empty entries
    train_count = len([x for x in train_data if x])
    test_count = len([x for x in test_data if x])
    
    print(f"Split completed (9:1 ratio, no validation): Train={train_count}, Test={test_count}")
        
###########################################################################

def read_data_new(path):
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
    return(data, interactions, user_num, item_num)

def remove_cold_items(data_path):
    cold_thre = 10  # to avoid cold/cool item (items with less than `cold_thre' records)
    path = data_path + '/train.json'
    dataset = read_data_new(path)
    data = dataset[0]
    interactions = dataset[1]
    user_num = dataset[2]
    item_num = dataset[3]
    
    max_user_id = max(interaction[0] for interaction in interactions)
    user_degrees = [0] * (max_user_id + 1)
    for interaction in interactions:
        user_id = interaction[0]
        user_degrees[user_id] += 1
    
    updated_interactions = []
    for interaction in interactions:
        user_id = interaction[0]
        if user_degrees[user_id] >= cold_thre:
            updated_interactions.append(interaction)
    
    updated_user_num = len(set(interaction[0] for interaction in updated_interactions))
    
    user_id_mapping = {user_id : new_id for new_id, user_id in enumerate(set(interaction[0] for interaction in updated_interactions))}
    re_indexed_interactions = []
    for interaction in updated_interactions:
        user_id = interaction[0]
        re_indexed_interactions.append((user_id_mapping[user_id], interaction[1]))
    
    max_item_id = max(interaction[1] for interaction in re_indexed_interactions)
    item_degrees = [0] * (max_item_id + 1)
    
    for interaction in re_indexed_interactions:
        item_id = interaction[1] 
        item_degrees[item_id] += 1
    
    final_interactions = []
    for interaction in re_indexed_interactions:
        item_id = interaction[1]
        if item_degrees[item_id] >= cold_thre:
            final_interactions.append(interaction)
    
    updated_item_num = len(set(interaction[1] for interaction in final_interactions))
        
    return cold_thre, data, final_interactions, updated_user_num, updated_item_num

def save_data(path):
    cold, data, interactions, user_num, item_num = remove_cold_items(path)
    
    print('cold_thre: ', cold)
    print('user_num: ', user_num)
    print('item_num: ', item_num)
    print('interaction_num: ', len(interactions))
    
    data_json = []
    for user in range(user_num):
        data_json.append([])
        
    for interaction in interactions:
        user_id = interaction[0]
        item_id = interaction[1]
        data_json[user_id].append(item_id)
        
    with open((path + '/train_data.json'), 'w') as train_file:
        json.dump(data_json, train_file)
    
# save_data('Gowalla/')
split_data_by_user('Yelp2018/')