#!/usr/bin/env python
# coding: utf-8

# In[51]:

import os

# Set environment variables before importing any relevant libraries
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'


from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from RuleTree.ensemble.RuleForestClassifier import RuleForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import itertools
import pandas as pd
import numpy as np
import copy

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, 
    f1_score, precision_score, recall_score, 
    classification_report
)
import time


# In[52]:


# In[4]:



# In[4]:
tabular = ['ionosphere', 'algerian_forest_fires_new','yeast_new','sonar', 'compas_new', 'german_onehot', 'twonorm', 'lrs', 'vertebral_column', 'iris_new', 'wine_new', 'diva', 'breast',  'steel_plates_faults', 'ecoli', 'heloc_new', 'page_blocks_binary_new','spambase_new','magic_new', 'house16_new']

images = ['doctoral','MNIST', 'cifar10', 'catsdogs', 'waterbirds',
          'oxfordpets', 'food101', 'organamnist', 'bloodmnist', 'SVHN','stanfordcars']

time_series = ['Yoga', 'StarLightCurves', 'ChlorineConcentration', 'SmallKitchenAppliances', 
               'SharePriceIncrease', 'ElectricDevices', 'GunPoint', 'WormsTwoClass', 'ECG5000', 'Wafer']

text = ['timschopf_medical_abstracts','vicunaspotify', 'palmTED',
        'tennisgpt','LiarPantsOnFire','polarity_dataset_v2.0_pang_cs_cornell_embed']
        #'imdb_new', 'news_new','lyrics_proc_embed_new']


#text = ['timschopf_medical_abstracts','vicunaspotify', 'palmTED',
#        'tennisgpt','LiarPantsOnFire','polarity_dataset_v2.0_pang_cs_cornell_embed',
#       'tdavidson_hate_speech_offensive_embed','imdb', 'news','lyrics_proc_embed']


already_split = images + time_series + text[0:-1]

all_datasets =  tabular + time_series + text + images

all_datasets = ['doctoral', 'yeast_new', 'algerian_forest_fires_new','iris_new','wine_new','page_blocks_binary_new','spambase_new','compas_new','magic_new','house16_new'] + ['polarity_dataset_v2.0_pang_cs_cornell_embed'] + images


all_datasets = text


results_folder = "datasets/RANDOM_FOREST_DEFAULT_testing"

for name in all_datasets:
    dataset_name = name
    file_path = os.path.join(results_folder, f"mstz_{dataset_name}_results_simple.csv")
    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping and waiting for 0.2 seconds...")
        time.sleep(0.2)
        
        # Check if dataset_name is either bloodmnist or organamnist
        if dataset_name in ['bloodmnist', 'organamnist']:
            continue
            #print(f"Processing {dataset_name} despite the file existing.")
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
    
    
    
    # In[55]:
    
    
    # Initialize StratifiedKFold
    scaler = StandardScaler()
    folds = []
    folds_proximity_matrix = []
    metric = 'euclidean'
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #X_train_matrix = pairwise_distances(X_train, metric=metric)
    X_train_matrix = []
    
    folds.append((X_train, X_test, y_train, y_test))
    folds_proximity_matrix.append(X_train_matrix)
        
    
    # In[56]:
    
    
    from RuleTree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
    from RuleTree.stumps.instance_stumps import * 
    
    
    # In[57]:
    
    
    pt_stump = pt_stump_call()
    dt_stump = dt_stump_call()
    obl_stump = obl_stump_call()
    obl_pt_stump = obl_pt_stump_call()
    multi_pt_stump = multi_pt_stump_call()
    multi_obl_pt_stump = multi_obl_pt_stump_call()
    
    
    # In[58]:
    
    
    rt = RuleForestClassifier(distance_measure='euclidean', 
                            base_stumps = [pt_stump,obl_pt_stump], 
                            stump_selection = 'best', 
                            random_state = 42, 
                            max_depth = 2,
                            n_estimators = 10,
                            max_samples=0.2,
                            max_features=0.2,
                            bootstrap = True, #can pick the same sample more then once
                            bootstrap_features = True, #can pick the same feat more then once
                            )
    
   # rt.fit(X_train,y_train)
    
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
    
    #estimators = rt.estimators_
   # data = [get_stumps(x) for x in estimators]
    
    
    def get_stumps_forest(data):
        result = defaultdict(int)
        for entry in data:
            for key, value in entry.items():
                result[key] += value
        result = dict(result)
    
        return result
            
    
    from collections import defaultdict
    
  #  result = get_stumps_forest(data)
    
    
    # In[59]:
    
    
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
    
    def calculate_averages(data):
        # Initialize a dictionary to store sums
        totals = {key: 0 for key in data[0].keys()}
        count = len(data)  # Total number of dictionaries
    
        # Calculate the sum for each key
        for entry in data:
            for key, value in entry.items():
                totals[key] += value
    
        # Calculate the averages
        averages = {key: totals[key] / count for key in totals.keys()}
        return averages
    
    
    def one_hot_encode_res(X, estimators):
        dictz = {'Rl': 0, 'Rr': 1}  # Mapping dictionary
        X_paths = []
    
        for model in estimators:
            labels = [dictz.get(x, -1) for x in model.apply(X)]  # Handle missing mappings with default
            X_paths.append(labels)
         #   print('Unique labels:', set(labels))
    
        return pd.DataFrame(np.array(X_paths).T)  # Create DataFrame from transposed array
    
    
    # In[60]:
    
    
    # Extract pivot information for all estimators
   # data_pivots = [info_pivots(x.get_pivots()) for x in estimators]
   # data_pivots = [{ 'n_' + k: len(v) for k, v in  data.items()} for data in data_pivots]
    
   # calculate_averages(data_pivots)
    
    
    # In[61]:
    
    
   # data_pivots
    
    
    # In[62]:
    
    
    pt_stump = pt_stump_call()
    dt_stump = dt_stump_call()
    obl_pt_stump = obl_pt_stump_call()
    multi_pt_stump = multi_pt_stump_call()
    multi_obl_pt_stump = multi_obl_pt_stump_call()
    
    
    # In[71]:
    
    
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

    all_stumps = [[dt_stump],[obl_stump],[obl_stump,dt_stump]]
    
    # Update the param_grid to use dynamically generated combinations
    
    """
    param_grid = {
        'max_depth': [2, 3, 4],
        'distance_measure': [metric],
        'random_state': [42],
        'base_stumps': all_stumps,
        'stump_selection': ['best'],
        'prune_useless_leaves': [True],
        'n_estimators' : [30,50,100],
        'max_features' : [0.3, 0.5, 0.7, 'sqrt'],
        'max_samples' : [0.3, 0.5, 0.7, 1.0],
        'bootstrap_features' : [True],
        'bootstrap' : [True],
        
    }
    
    """
    
    param_grid = {
        'max_depth': [2, 3, 4],
        'distance_measure': [metric],
        'random_state': [42],
        'base_stumps': all_stumps,
        'stump_selection': ['best'],
        'prune_useless_leaves': [True],
        'n_estimators' : [100],
        'max_features' : ['sqrt'],
        'max_samples' : [None],
        'bootstrap_features' : [False],
        'bootstrap' : [False],
        
    }
    
    
    
    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    param_combinations = list(itertools.product(*values))
    
    
    # In[72]:
    
    
    print(len(param_combinations))
    
    
    # In[66]:
    
    
    # Fit the model
    #rt.fit(X_train_fold, y_train_fold)
    
    #info_pivots(rt.get_pivots())
    
    
    # In[67]:
    
    
    from sklearn.linear_model import LogisticRegression
    extra_classifiers = {}
    
    # Add KNN classifiers with fixed neighbors
    for p in [0.05, 0.10, 0.15, 0.2, 0.25 , 0.3, 0.35, 0.4, 0.45]:
        extra_classifiers[p] =  LogisticRegression(max_iter = 10000)
    
    
    # In[68]:
    
    
    extra_classifiers
    
    
    # In[69]:
    
    
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
    
    def n_est_update(results_df):
        val = results_df['n_estimators'].iloc[0]
        n_estimators_base = [val]
        for e, row in results_df.iterrows():
            if e == 0:
                continue
            additional = row['Additional Classifier']
            
            if additional == 'NO':
                val = row['n_estimators']
            n_estimators_base += [val]
            
        n_estimators_new = n_estimators_base
        n_estimators_real = list(results_df.n_estimators)
    
        results_df['n_estimators'] = n_estimators_new
        results_df['n_estimators_real'] = n_estimators_real
        
        columns = list(results_df.columns)
        columns.remove('n_estimators_real')
        n_estimators_index = columns.index('n_estimators')
        columns.insert(n_estimators_index + 1, 'n_estimators_real')
        results_df = results_df[columns]
        
    
        return results_df

    
    import pandas as pd
    
    # Initialize a list to store results
    results = []
    
    
    from joblib import parallel_backend
    
    with parallel_backend('loky', n_jobs=4):
        
        # Iterate over folds and corresponding distance matrices
        for fold_idx, (fold_tuple, fold_matrix) in enumerate(zip(folds, folds_proximity_matrix)):
            X_train_fold, X_test_fold, y_train_fold, y_test_fold = fold_tuple
            X_train_fold_matrix = fold_matrix
        
            for e, combination in enumerate(param_combinations):
                print(f'{e}-{fold_idx}')
            
                # Create dictionary of parameters
                params = dict(zip(keys, combination))
        
                # Initialize the classifier with the current parameter set and distance matrix
                rt = RuleForestClassifier(**params, n_jobs = 4)
              
                
        
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
                
                #stump_counts = get_stumps(rt)
                data = [get_stumps(x) for x in rt.estimators_]
                stump_counts = get_stumps_forest(data)
                
                #data_pivots = [info_pivots(x.get_pivots()) for x in rt.estimators_]
                #data_pivots = [{ 'n_' + k: len(v) for k, v in  data.items()} for data in data_pivots]
                #data_pivots = calculate_averages(data_pivots)
                data_pivots = {'n_discriminatives': 0,
                                  'n_descriptives': 0,
                                  'n_used': 0,
                                  'n_candidates': 0}
        
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
                    "n_estimators" : len(rt.estimators_),
                    "n_estimators_real" : len(rt.estimators_),
                    **data_pivots,
                    **metrics,
                    **stump_counts
                }
                
                results.append(result_entry) 
                
                
                n_old_estimators = len(rt.estimators_)
                old_estimators = copy.deepcopy(rt.estimators_)
        
                for p, clf in extra_classifiers.items():
                    balanced_stumps = {}
                    
                    for est in old_estimators:
                        bal_stumps = est.get_balanced_stumps(p = p)
                        trees = est.stumps_to_trees(bal_stumps)
                        
                        for k, v in trees.items():
                            balanced_stumps[k] =  v

                    
                    rt.estimators_ = list(balanced_stumps.values())
                    
                    start_predict_extra = time.process_time()
                    y_pred = rt.predict(X_test_fold)
                    end_predict_extra = time.process_time()
                    predict_time_extra = end_predict_extra - start_predict_extra
        
                    data = [get_stumps(x) for x in rt.estimators_]
                    
                    stump_counts = get_stumps_forest(data)
                    #data_pivots = [info_pivots(x.get_pivots()) for x in rt.estimators_]
                    #data_pivots = [{ 'n_' + k: len(v) for k, v in  data.items()} for data in data_pivots]
                    data_pivots = {}
                   # print(p)
                   # print(data_pivots)
        
                    if len(data_pivots) < 1:
                        data_pivots = {'n_discriminatives': 0,
                                  'n_descriptives': 0,
                                  'n_used': 0,
                                  'n_candidates': 0}
                    else:
                        data_pivots = calculate_averages(data_pivots)
        
                    metrics = evaluate_metrics(y_test_fold, y_pred)
                    
                    result_entry = {
                                    **params,
                                    "Fold": fold_idx + 1,
                                    "Fit Time (s)": fit_time,
                                    "Predict Time (s)": predict_time,
                                    "Additional Classifier" : f'{p}_balance_forest',
                                    "Additional Classifier Fit Time (s)" : 0.0,
                                    "Additional Classifier Predict Time (s)" : predict_time_extra,
                                    "n_estimators" : n_old_estimators,
                                    "n_estimators_real" : len(rt.estimators_),
                                    **data_pivots,
                                    **metrics,
                                    **stump_counts
                                }
                                                
                    results.append(result_entry)
        
                    if not len(rt.estimators_) < 1:
            
                        X_train_encoded = one_hot_encode_res(X_train_fold, rt.estimators_)
                        X_test_encoded = one_hot_encode_res(X_test_fold, rt.estimators_)
                        
                        # Combine encoded data
                        df_full = pd.concat([X_train_encoded, X_test_encoded], axis=0, ignore_index=True)
                        
                        # Deduplicate columns in the combined dataset
                        df_full = df_full.loc[:, ~df_full.T.duplicated()]
                        
                        # Split back into deduplicated X_train and X_test
                        X_train_reduced = df_full.iloc[:len(X_train_encoded), :]
                        X_test_reduced = df_full.iloc[len(X_train_encoded):, :]
            
                        LR = LogisticRegression(max_iter = 1000)
            
                        start_fit_extra = time.process_time()
                        LR.fit(X_train_reduced, y_train_fold)
                        end_fit_extra = time.process_time()
                        fit_time_extra = end_fit_extra - start_fit_extra
                        
                        start_predict_extra = time.process_time()
                        y_pred = LR.predict(X_test_reduced)
                        end_predict_extra = time.process_time()
                        predict_time_extra = end_predict_extra - start_predict_extra
                                    
                        metrics = evaluate_metrics(y_test_fold, y_pred)
                        
                        result_entry = {
                                        **params,
                                        "Fold": fold_idx + 1,
                                        "Fit Time (s)": fit_time,
                                        "Predict Time (s)": predict_time,
                                        "Additional Classifier" : f'{p}_balance_forest_LR',
                                        "Additional Classifier Fit Time (s)" : start_fit_extra,
                                        "Additional Classifier Predict Time (s)" : predict_time_extra,
                                        "n_estimators" : n_old_estimators,
                                        "n_estimators_real" : X_train_reduced.shape[1],
                                        **data_pivots,
                                        **metrics,
                                        **stump_counts
                                    }
                                                    
                        results.append(result_entry)
        
                    
                    else:
                        metrics = metric_dummy(y_test_fold,y_pred)
                        result_entry = {
                                        **params,
                                        "Fold": fold_idx + 1,
                                        "Fit Time (s)": fit_time,
                                        "Predict Time (s)": predict_time,
                                        "Additional Classifier" : f'{p}_balance_forest_LR',
                                        "Additional Classifier Fit Time (s)" : 0.0,
                                        "Additional Classifier Predict Time (s)" : predict_time_extra,
                                        "n_estimators" : n_old_estimators,
                                        "n_estimators_real" : 0.0,
                                        **data_pivots,
                                        **metrics,
                                        **stump_counts
                                    }
                                                    
                        results.append(result_entry)
                
    
    results_df = pd.DataFrame(results)
   
    
    
    
    # In[44]:
    
    
    results[-1]
    
    
    # In[45]:
    
    
    results_df.base_stumps = [str(x) for x in results_df.base_stumps]
    
    
    # In[26]:
    
    
    results_df['Additional Classifier Fit Time (s)']
    
    
    
    
    # In[27]:
    
    
    results_df.to_csv(f'datasets/RANDOM_FOREST_DEFAULT_testing/mstz_{dataset_name}_results_simple.csv', index = False)

    standard_forest = results_df.groupby(
    ['max_depth', 'distance_measure', 'random_state', 'base_stumps','stump_selection', 'prune_useless_leaves', 
     'n_estimators', 'max_features', 'max_samples', 'Additional Classifier','bootstrap_features','bootstrap']
    ).agg(['mean', 'std'])
    
    standard_forest.columns = ['_'.join(col).strip() for col in standard_forest.columns.values]
    # Reset the index for a cleaner output
    standard_forest = standard_forest.reset_index()
    standard_forest.sort_values(by=['Weighted F1 Score_mean'], ascending=False)
    
    grouped_results = standard_forest
    
    grouped_results.to_csv(f'datasets/RANDOM_FOREST_DEFAULT_testing/mstz_{dataset_name}_grouped_simple.csv', index = False)
    
    
    # In[28]:
    """
    
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
    
    
    grouped_results.to_csv(f'datasets/RF_kfold_res/mstz_{dataset_name}_grouped.csv', index = False)
    
    
    # In[ ]:
    
    
    """
    
    
    # In[ ]:
    
    
    
    
