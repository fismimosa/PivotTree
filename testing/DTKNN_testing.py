#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os

# Set environment variables before importing any relevant libraries
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['GOTO_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'



from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import itertools
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, 
    f1_score, precision_score, recall_score, 
    classification_report
)
import time


# In[4]:



tabular = ['ionosphere', 'algerian_forest_fires_new','yeast_new', 'magic_new', 'sonar', 'compas_new', 'house16_new', 'german_onehot', 'spambase_new', 'twonorm', 'lrs', 'vertebral_column', 'iris_new', 'wine_new', 'diva', 'breast',  'steel_plates_faults', 'ecoli', 'heloc_new', 'page_blocks_binary_new']

images = ['doctoral','MNIST', 'cifar10', 'catsdogs', 'waterbirds',
          'oxfordpets', 'food101', 'organamnist', 'bloodmnist', 'SVHN','stanfordcars']

time_series = ['Yoga', 'StarLightCurves', 'ChlorineConcentration', 'SmallKitchenAppliances', 
               'SharePriceIncrease', 'ElectricDevices', 'GunPoint', 'WormsTwoClass', 'ECG5000', 'Wafer']

text = ['timschopf_medical_abstracts','vicunaspotify', 'palmTED',
        'tennisgpt','polarity_dataset_v2.0_pang_cs_cornell_embed',
        'imdb_new', 'news_new','lyrics_proc_embed_new']


#text = ['timschopf_medical_abstracts','vicunaspotify', 'palmTED',
#        'tennisgpt','LiarPantsOnFire','polarity_dataset_v2.0_pang_cs_cornell_embed',
#       'tdavidson_hate_speech_offensive_embed','imdb', 'news','lyrics_proc_embed']


already_split = images + time_series + text[0:-1]

all_datasets = tabular + images + time_series + text



results_folder = "datasets/standard_testing_res"

for name in all_datasets:
    dataset_name = name
    file_path = os.path.join(results_folder, f"mstz_{dataset_name}_results.csv")
    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping and waiting for 0.2 seconds...")
        time.sleep(0.2)
        
        # Check if dataset_name is either bloodmnist or organamnist
        if dataset_name in ['bloodmnist', 'organamnist']:
            print(f"Processing {dataset_name} despite the file existing.")
            continue
        else:
            continue
        


    print(dataset_name)

    
    if dataset_name in already_split:
        try:
            dataset_train = pd.read_csv(f'datasets/split_datasets/mstz_{dataset_name}_train_embedding.csv')
            dataset_test = pd.read_csv(f'datasets/split_datasets/mstz_{dataset_name}_test_embedding.csv')
        except:
            dataset_train = pd.read_csv(f'datasets/split_datasets/mstz_{dataset_name}_train.csv')
            dataset_test = pd.read_csv(f'datasets/split_datasets/mstz_{dataset_name}_test.csv')
            
            
        X_train = dataset_train.drop(columns = 'label').values
        y_train = np.array(dataset_train.label)
    
        X_test = dataset_test.drop(columns = 'label').values
        y_test = np.array(dataset_test.label)
    
    
        print(X_train.shape, X_test.shape)
    
    else:
    
        dataset = pd.read_csv(f'datasets/mstz_{dataset_name}.csv')
        X = dataset.drop(columns = 'label').values
        y = np.array(dataset.label)
        print(X.shape)
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.3, 
                                                            random_state=42)
    
    
        
            
       
    
    
    
    # In[5]:
    
    
    scaler = StandardScaler()
    folds = []
    folds_proximity_matrix = []
    metric = 'euclidean'
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_matrix = pairwise_distances(X_train, metric=metric)
    
    folds.append((X_train, X_test, y_train, y_test))
    folds_proximity_matrix.append(X_train_matrix)
    
    
    # In[8]:
    
    
    from RuleTree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
    from RuleTree.stumps.instance_stumps import * 
    
    
    # In[9]:
    
    
    pt_stump = pt_stump_call()
    dt_stump = dt_stump_call()
    obl_pt_stump = obl_pt_stump_call()
    multi_pt_stump = multi_pt_stump_call()
    multi_obl_pt_stump = multi_obl_pt_stump_call()
    
    
    # In[10]:
    
    rt = RuleTreeClassifier(distance_measure='euclidean', base_stumps = [dt_stump], stump_selection = 'best', random_state = 0, 
                            max_depth = 3)
    rt.fit(X_train,y_train)
    def get_stumps(rt, current_node=None, stump_dict=None):
        # Initialize the stump dictionary if not provided
        if stump_dict is None:
            stump_dict = {}
    
        # Start from the root if current_node is not provided
        if current_node is None:
            current_node = rt.root
    
        # Process the current node
        if not current_node.is_leaf() and current_node.stump:
            stump_name = current_node.stump.__class__.__module__.split('.')[-1]
            # Dynamically initialize the count for the stump type if not already present
            stump_key = f"n_{stump_name}"
            stump_dict[stump_key] = stump_dict.get(stump_key, 0) + 1
    
        # Recurse into left and right child nodes if they exist
        if current_node.node_l:
            get_stumps(rt, current_node.node_l, stump_dict)
        if current_node.node_r:
            get_stumps(rt, current_node.node_r, stump_dict)
    
        return stump_dict
    
    
    # Example Usage
    stump_counts = get_stumps(rt)
    print(stump_counts)
    
    
    
    # In[11]:
    
    
    pt_stump = pt_stump_call()
    obl_stump = obl_stump_call()
    dt_stump = dt_stump_call()
    obl_pt_stump = obl_pt_stump_call()
    multi_pt_stump = multi_pt_stump_call()
    multi_obl_pt_stump = multi_obl_pt_stump_call()
        
    
    # In[12]:
    
    
    import itertools
    
    # Define the stumps
    pt_stump = pt_stump_call()
    dt_stump = dt_stump_call()
    obl_pt_stump = obl_pt_stump_call()
    multi_pt_stump = multi_pt_stump_call()
    multi_obl_pt_stump = multi_obl_pt_stump_call()
    
    # Create a list of all possible stumps
    all_stumps = [pt_stump, obl_pt_stump, multi_pt_stump, multi_obl_pt_stump]
    all_stumps = [[pt_stump],[obl_pt_stump],[multi_pt_stump],[multi_obl_pt_stump], all_stumps]
    
    
    param_grid = {
        'max_depth': [1],
        'distance_measure': [metric],
        'random_state': [42],
        'base_stumps': [dt_stump],
        'stump_selection': ['best'],
        'prune_useless_leaves': [True],
    }
    
    
    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    param_combinations = list(itertools.product(*values))
    
    
    # In[13]:
    
    
    def info_pivots(pivots):
        # Initialize lists to store data
        discriminatives = []
        descriptives = []
        used = []
        candidates = []
        
        # Iterate through the pivot data
        for k, v in pivots.items():
            discriminatives += list(v.get('discriminatives', []))  # Extract discriminatives
            descriptives += list(v.get('descriptives', []))        # Extract descriptives
            used += list(v.get('used', [])) if isinstance(v.get('used', []), list) else [v.get('used', [])]  # Ensure `used` is always a list
            candidates += list(v.get('candidates', []))            # Extract candidates
        
        # Ensure unique values in each list
        discriminatives = list(set(discriminatives))
        descriptives = list(set(descriptives))
        used = list(set(used))
        candidates = list(set(candidates))
    
        # Return a dictionary with the appropriate keys
        return {
            'discriminatives': discriminatives,
            'descriptives': descriptives,
            'used': used,
            'candidates': candidates
        }
    
    
    # In[14]:
    
    
    # Fit the model
    #rt.fit(X_train_fold, y_train_fold)
    
    #info_pivots(rt.get_pivots())
    
    
    # In[15]:
    
    
    from sklearn.neighbors import KNeighborsClassifier
    depths = [2, 3, 4, 5, 6]
    
    extra_classifiers = {}
    for depth in depths:
        extra_classifiers[f'standard_dt_{depth}'] = RuleTreeClassifier(
            max_depth=depth, min_samples_leaf=1, min_samples_split=2, random_state=42, base_stumps = [dt_stump], stump_selection = 'best'
        )
        extra_classifiers[f'standard_obldt_{depth}'] = RuleTreeClassifier(
            max_depth=depth, min_samples_leaf=1, min_samples_split=2, random_state=42, base_stumps = [obl_stump], stump_selection = 'best'
        )
    
    # Add KNN classifiers with fixed neighbors
    for k in [1, 3, 5]:
        extra_classifiers[f'standard_knn_{k}'] = KNeighborsClassifier(n_neighbors=k)
    
    
    # In[16]:
    
    
    import pandas as pd
    import time
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, pairwise_distances
    )
    
    def evaluate_metrics(y_true, y_pred):
        """Compute evaluation metrics."""
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Weighted F1 Score": f1_score(y_true, y_pred, average="weighted"),
            "Macro F1 Score": f1_score(y_true, y_pred, average="macro"),
            "Weighted Precision": precision_score(y_true, y_pred, average="weighted"),
            "Macro Precision": precision_score(y_true, y_pred, average="macro"),
            "Weighted Recall": recall_score(y_true, y_pred, average="weighted"),
            "Macro Recall": recall_score(y_true, y_pred, average="macro"),
        }
    
    def metric_dummy(y_true,y_pred):
        return {
            "Accuracy": 0.0,
            "Balanced Accuracy": 0.0,
            "Weighted F1 Score": 0.0,
            "Macro F1 Score": 0.0,
            "Weighted Precision":0.0,
            "Macro Precision": 0.0,
            "Weighted Recall": 0.0,
            "Macro Recall": 0.0,
        }
    
    
    # In[15]:
    
    
    import pandas as pd
    
    # Initialize a list to store results
    results = []
    
    # Iterate over folds and corresponding distance matrices
    for fold_idx, (fold_tuple, fold_matrix) in enumerate(zip(folds, folds_proximity_matrix)):
        X_train_fold, X_test_fold, y_train_fold, y_test_fold = fold_tuple
        X_train_fold_matrix = fold_matrix
    
        for combination in param_combinations:
        
            # Create dictionary of parameters
            params = dict(zip(keys, combination))
    
            # Initialize the classifier with the current parameter set and distance matrix
            rt = RuleTreeClassifier(**params, distance_matrix=X_train_fold_matrix)
    
            # Measure fitting time
            start_fit = time.process_time()
            rt.fit(X_train_fold, y_train_fold)
            end_fit = time.process_time()
            fit_time = end_fit - start_fit
    
            # Measure prediction time
            start_predict = time.process_time()
            y_pred = rt.predict(X_test_fold)
            end_predict = time.process_time()
            predict_time = end_predict - start_predict
            
            stump_counts = get_stumps(rt)
    
           # dict_pivs = info_pivots(rt.get_pivots())
    
            # Compute evaluation metrics
            metrics = evaluate_metrics(y_test_fold, y_pred)
        
            # Create a result entry for this fold and parameter combination
            result_entry = {
                **params,
                "Fold": fold_idx + 1,
                "Fit Time (s)": fit_time,
                "Predict Time (s)": predict_time,
                "Additional Classifier" : 'NO',
                "Additional Classifier fit time" : 0.0,
                "Additional Classifier predict time" : 0.0,
                #"n_discriminatives" : len(dict_pivs['discriminatives']),
                #"n_descriptives" : len(dict_pivs['descriptives']),
                #"n_candidates" : len(dict_pivs['candidates']),
                #"n_used" : len(dict_pivs['used']),
                **metrics,
                **stump_counts
            }
            
            results.append(result_entry)      
    
            for _ in range(1):
                #X_train_reduced = pairwise_distances(X_train_fold, X_train_fold[value] , metric = metric)
                #X_test_reduced = pairwise_distances(X_test_fold, X_train_fold[value] , metric = metric)
                
                for extra_clf_name, extra_clf in extra_classifiers.items():
                  #  if 'knn_' in extra_clf_name:
                        
                            start_fit_extra = time.process_time()
                            extra_clf.fit(X_train_fold, y_train_fold)
                            end_fit_extra = time.process_time()
                            fit_time_extra = end_fit_extra - start_fit_extra
        
                            start_predict_extra = time.process_time()
                            y_pred = extra_clf.predict(X_test_fold)
                            end_predict_extra = time.process_time()
                            predict_time_extra = end_predict_extra - start_predict_extra
                            
                            metrics = evaluate_metrics(y_test_fold, y_pred)
                            
                            result_entry = {
                                **params,
                                "Fold": fold_idx + 1,
                                "Fit Time (s)": fit_time,
                                "Predict Time (s)": predict_time,
                                "Additional Classifier" : f'{extra_clf_name}',
                                "Additional Classifier Fit Time (s)" : start_fit_extra,
                                "Additional Classifier Predict Time (s)" : predict_time_extra,
                                #"n_discriminatives" : len(dict_pivs['discriminatives']),
                                #"n_descriptives" : len(dict_pivs['descriptives']),
                                #"n_candidates" : len(dict_pivs['candidates']),
                                #"n_used" : len(dict_pivs['used']),
                                **metrics,
                                **stump_counts
                            }
                            
                            results.append(result_entry)
                        
                    
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    
    
    
    # In[16]:
    
    
    results_df.base_stumps = [str(x) for x in results_df.base_stumps]
    
    
    results_df['Additional Classifier Fit Time (s)']
    
    
    # In[27]:
    
    
    results_df.to_csv(f'datasets/standard_testing_res/mstz_{dataset_name}_results.csv', index = False)
    
    
    # In[28]:
    
    
    # Group by 'Fold' and all the specified columns and calculate mean and std for numeric columns
    grouped_results = results_df.groupby(
        ['max_depth', 'distance_measure', 'random_state', 'base_stumps', 
         'stump_selection', 'prune_useless_leaves', 'Additional Classifier']
    ).agg(['mean', 'std'])
    
    # Flatten the multi-level columns
    grouped_results.columns = ['_'.join(col).strip() for col in grouped_results.columns.values]
    
    # Reset the index for a cleaner output
    grouped_results = grouped_results.reset_index()
    
    
    
    # In[29]:
    
    
    grouped_results.sort_values(by=['Weighted F1 Score_mean'], ascending=False)
    
    
    # In[22]:
    
    
    grouped_results.to_csv(f'datasets/standard_testing_res/mstz_{dataset_name}_grouped.csv', index = False)
    
    
    
    
