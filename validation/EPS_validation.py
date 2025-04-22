#!/usr/bin/env python
# coding: utf-8

# In[100]:


import os

# Set environment variables before importing any relevant libraries
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


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

from alibi.prototypes import ProtoSelect
from alibi.utils.kernel import EuclideanDistance

# In[4]:

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

all_datasets = tabular  + time_series + text + images

results_folder = "datasets/eps_validation"

for name in all_datasets:
    dataset_name = name
    file_path = os.path.join(results_folder, f"mstz_{dataset_name}_results.csv")
    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping and waiting for 0.2 seconds...")
        time.sleep(0.2)
        
        # Check if dataset_name is either bloodmnist or organamnist
        if dataset_name in ['bloodmnist', 'organamnist']:
            print(f"Processing {dataset_name} despite the file existing.")
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
    
    
    
    
    # In[106]:
    
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    folds = []
    folds_proximity_matrix = []
    quantiles_per_fold = []
    metric = 'euclidean'
    
    
    quantile_range = np.arange(0.00, 0.54, 0.02)  # 2%, 4%, ..., 50%
    
    
    for fold_idx, (train_indices, test_indices) in enumerate(skf.split(X_train, y_train)):
        # Define the training and test set for the current fold
        X_train_fold, X_test_fold = X_train[train_indices], X_train[test_indices]
        y_train_fold, y_test_fold = y_train[train_indices], y_train[test_indices]
        
        # Perform standard scaling on X_train_fold and apply the transformation to X_test_fold
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_test_fold_scaled = scaler.transform(X_test_fold)

        X_train_fold = X_train_fold_scaled
        X_test_fold = X_test_fold_scaled
        
        
        # Compute the proximity matrix (distance matrix) for the scaled training data
        X_train_fold_matrix = pairwise_distances(X_train_fold_scaled, metric=metric)
        
        # Store the fold data and proximity matrix
        folds.append((X_train_fold, X_test_fold, y_train_fold, y_test_fold))
        folds_proximity_matrix.append(X_train_fold_matrix)
        
        distances = X_train_fold_matrix.flatten()  # Flatten the matrix to a 1D array
        quantiles = np.quantile(distances, quantile_range)  # Compute desired quantiles
        quantiles = np.array([np.min(distances)] + list(quantiles))
        quantiles_per_fold.append(quantiles)
        
    
    # In[107]:
    
    
    from RuleTree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
    from RuleTree.stumps.instance_stumps import * 
    
    pt_stump = pt_stump_call()
    dt_stump = dt_stump_call()
    obl_stump = obl_stump_call()
    obl_pt_stump = obl_pt_stump_call()
    multi_pt_stump = multi_pt_stump_call()
    multi_obl_pt_stump = multi_obl_pt_stump_call()
    
    
    # In[108]:
    
    
    from sklearn.neighbors import KNeighborsClassifier
    depths = [2, 3, 4, 5, 6]
    
    extra_classifiers = {}
    for depth in depths:
        extra_classifiers[f'dt_{depth}'] = RuleTreeClassifier(
            max_depth=depth, min_samples_leaf=1, min_samples_split=2, random_state=42, base_stumps = [dt_stump], stump_selection = 'best'
        )
        extra_classifiers[f'obldt_{depth}'] = RuleTreeClassifier(
            max_depth=depth, min_samples_leaf=1, min_samples_split=2, random_state=42, base_stumps = [obl_stump], stump_selection = 'best'
        )
        
    
    # Add KNN classifiers with fixed neighbors
    for k in [1, 3, 5]:
        extra_classifiers[f'knn_{k}'] = KNeighborsClassifier(n_neighbors=k)
    
    
    # In[109]:
    
    
    extra_classifiers
    
    """
    # In[110]:
    
    
    summariser = ProtoSelect(kernel_distance=EuclideanDistance(), eps=0.1,)
    
    
    # In[111]:
    
    
    summariser
    
    
    # In[112]:
    
    
    dir(summariser)
    summariser.batch_size
    
    
    # In[113]:
    
    
    summariser = ProtoSelect(kernel_distance=EuclideanDistance(), eps=0.1)
    
    ress = []
    for x in range(10):
        start_fit = time.process_time()
        summariser = summariser.fit(X_train_fold, y_train_fold)
        fit_time = time.process_time() - start_fit
        summary = summariser.summarise(len(X_train_fold))
        indices = summary.prototype_indices
        ress.append((indices))
    
    
    # In[114]:
    
    
    #[len(x) for x in ress],
    
    ress
    
    
    # In[115]:
    
    
    summary = summariser.summarise(len(X_train_fold))
    
    
    # In[116]:
    
    
    summary
    
    """
    
    # In[117]:
    
    
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
    
    
    # In[ ]:
    
    
    
    
    
    # In[ ]:
    
    
    results = []
    
    
    for fold_idx, (fold_tuple, fold_quantile) in enumerate(zip(folds, quantiles_per_fold)):
        X_train_fold, X_test_fold, y_train_fold, y_test_fold = fold_tuple
        quantiles = fold_quantile
        range_quant = [-1] + [x for x in list(quantile_range)]
       
        
        for e, eps in enumerate(quantiles):
            if eps == 0:
                continue
            fold_random_state = np.random.RandomState(seed=42)
            
            params = {'eps' : eps, 'quantile' : range_quant[e],  'distance_measure' : 'euclidean', 'random_state' :42}   
            summariser = ProtoSelect(kernel_distance=EuclideanDistance(), eps=eps)
            
            start_fit = time.process_time()
            summariser = summariser.fit(X_train_fold, y_train_fold)
            fit_time = time.process_time() - start_fit
    
            predict_time = 0.0 #as we are not performing prediciton but just finding the prots
    
            summary = summariser.summarise(num_prototypes=len(X_train_fold))
            summary_max_5 = summariser.summarise(num_prototypes=5)
            summary_max_10 = summariser.summarise(num_prototypes=10)
            summary_max_20 = summariser.summarise(num_prototypes=20)
    
            prototypes = {
                "full": summary.prototype_indices,
                "max_5": summary_max_5.prototype_indices,
                "max_10": summary_max_10.prototype_indices,
                "max_20": summary_max_20.prototype_indices
            }

            if len(prototypes['full']) < 1: #if no prots found
                   print('full empty')
                   continue
    
            for subset_key, prototype_indices in prototypes.items():
                X_train_reduced = pairwise_distances(X_train_fold, X_train_fold[prototype_indices] , metric = metric)
                X_test_reduced = pairwise_distances(X_test_fold, X_train_fold[prototype_indices] , metric = metric)
                
                for clf_name, clf in extra_classifiers.items():
                    
                    start_fit_extra = time.process_time()
                    clf.fit(X_train_reduced, y_train_fold)
                    end_fit_extra = time.process_time()
                    fit_time_extra = end_fit_extra - start_fit_extra
        
                    start_predict_extra = time.process_time()
                    y_pred = clf.predict(X_test_reduced)
                    end_predict_extra = time.process_time()
                    predict_time_extra = end_predict_extra - start_predict_extra
        
                    metrics = evaluate_metrics(y_test_fold, y_pred)
                        
    
                    result_entry = {
                           **params,
                           
                            "Fold": fold_idx + 1,
                            "Fit Time (s)": fit_time,
                            "Predict Time (s)": predict_time,
                            "Additional Classifier" : f'eps_{subset_key}_{clf_name}',
                            "Additional Classifier Fit Time (s)" : start_fit_extra,
                            "Additional Classifier Predict Time (s)" : predict_time_extra,
                            "n_prots_found" : len(prototypes['full']),
                            "n_prots_used": len(prototype_indices),
                            **metrics,
                          #  **stump_counts
                        }
                    
                    results.append(result_entry)
                    
    
                    
                    if 'knn_' in clf_name:
                        try:
                            # Fit classifier
                            start_fit_extra = time.process_time()
                            clf.fit(X_train_fold[prototype_indices], y_train_fold[prototype_indices])
                            fit_time_extra = time.process_time() - start_fit_extra
        
                            # Predict
                            start_predict_extra = time.process_time()
                            y_pred = clf.predict(X_test_fold)
                            predict_time_extra = time.process_time() - start_predict_extra
        
                            metrics = evaluate_metrics(y_test_fold, y_pred)
                        except Exception:
                            fit_time_extra, predict_time_extra = 0.0, 0.0
                            metrics = metric_dummy(y_test_fold, y_pred)
        
                        result_entry = {
                            **params,
                            "eps" : eps,
                            "Fold": fold_idx + 1,
                            "Fit Time (s)": fit_time,
                            "Predict Time (s)": predict_time,
                            "Additional Classifier": f'eps_{subset_key}_{clf_name}_feature_space',
                            "Additional Classifier Fit Time (s)": fit_time_extra,
                            "Additional Classifier Predict Time (s)": predict_time_extra,
                            "n_prots_found" : len(prototypes['full']),
                            "n_prots_used": len(prototype_indices),
                            **metrics,
                      #      **stump_counts
                        }
                        results.append(result_entry)
    
    
                            
                           
    
                    
    results_df = pd.DataFrame(results)
    
    
    # In[ ]:
    
    
    #results_df[results_df['Additional Classifier'] == 'eps_max_20_knn_1_feature_space']
    results_df
    
    results_df.to_csv(f'datasets/eps_validation/mstz_{dataset_name}_results.csv', index = False)
    
    
    # In[ ]:
    
    
    grouped_results = results_df.groupby(
            ['quantile', 'distance_measure'	, 'random_state',	'Additional Classifier']
        ).agg(['mean', 'std'])
    
    # In[ ]:
    
    
    # Flatten the multi-level columns
    grouped_results.columns = ['_'.join(col).strip() for col in grouped_results.columns.values]
    
    # Reset the index for a cleaner output
    grouped_results = grouped_results.reset_index()
    
    
    
    # In[18]:
    
    
    grouped_results.sort_values(by=['Weighted F1 Score_mean'], ascending=False)
    
    
    # In[ ]:
    
    
    grouped_results.to_csv(f'datasets/eps_validation/mstz_{dataset_name}_grouped.csv', index = False)
    
    
    
    
    
    
    # In[ ]:
    
    
    
    
    
    # In[ ]:
    
    
    
    
    
    # In[ ]:
    
    
    
    
    
    # In[ ]:
    
    
    
    
