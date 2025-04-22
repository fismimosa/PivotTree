#!/usr/bin/env python
# coding: utf-8

# In[128]:



# In[51]:
import os

# Set environment variables before importing any relevant libraries
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['GOTO_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'




import pandas as pd
import numpy as np
from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import itertools
import pandas as pd
import numpy as np
import random


from pyclustering.cluster.kmeans import kmeans
import numpy as np
from scipy.spatial.distance import cdist  # For efficient distance computation


from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, 
    f1_score, precision_score, recall_score, 
    classification_report
)
import time

from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer 
import numpy as np
import warnings
import kmedoids
np.warnings = warnings


# In[129]:


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


results_folder = "datasets/KME_MED_RIGHT_testing"

for name in all_datasets:
    dataset_name = name
    file_path = os.path.join(results_folder, f"mstz_{dataset_name}_results.csv")
    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping and waiting for 0.2 seconds...")
        time.sleep(0.2)
        
        # Check if dataset_name is either bloodmnist or organamnist
        if dataset_name in ['bloodmnist', 'organamnist']:
            continue
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
    
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    folds = []
    folds_proximity_matrix = []
    quantiles_per_fold = []
    metric = 'euclidean'
    
    
    k_range = np.arange(2,51,1)
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    folds = []
    folds_proximity_matrix = []
    quantiles_per_fold = []
    metric = 'euclidean'
    
    quantile_range = np.arange(0.02, 0.41, 0.02)  # 2%, 4%, ..., 40%
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_matrix = pairwise_distances(X_train, metric=metric)
    
    folds.append((X_train, X_test, y_train, y_test))
    folds_proximity_matrix.append(X_train_matrix)
  
        
    from RuleTree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
    from RuleTree.stumps.instance_stumps import * 
    
    pt_stump = pt_stump_call()
    dt_stump = dt_stump_call()
    obl_stump = obl_stump_call()
    obl_pt_stump = obl_pt_stump_call()
    multi_pt_stump = multi_pt_stump_call()
    multi_obl_pt_stump = multi_obl_pt_stump_call()
    
    
    
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
    
    
    # In[134]:
    
    
    extra_classifiers
    
    
    # In[135]:
    
    
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
    
    
    # In[143]:
    
    
    results = []
    
    for fold_idx, (fold_tuple, fold_matrix) in enumerate(zip(folds, folds_proximity_matrix)):
        X_train_fold, X_test_fold, y_train_fold, y_test_fold = fold_tuple
        X_train_fold_matrix = fold_matrix
        params = {'distance_measure' : 'euclidean', 'random_state' :42}   
        
        k_range = np.arange(2,51,1)
        
        for k in k_range:
            print(k, fold_idx)
            if k > len(X_train_fold):
                print('too high k, skip')
                continue
                
            fold_random_state = np.random.RandomState(seed=42)
            np.random.seed(42)
    
            initial_centers = kmeans_plusplus_initializer(X_train_fold,k).initialize(seed = 42)
            initial_indices = [np.where((X_train_fold == center).all(axis=1))[0][0] for center in initial_centers]
    
            kmeans_instance = kmeans(X_train_fold, initial_centers)
            #kmedoids_instance = kmedoids(X_train_fold, initial_indices)
    
            start_fit_kmeans = time.process_time()
            kmeans_instance.process()
            fit_time_kmeans = time.process_time() - start_fit_kmeans
            
            start_fit_kmedoids = time.process_time()
            #kmedoids_instance.process()
            
            kmedoids_instance = kmedoids.fasterpam(diss = X_train_fold_matrix, medoids= np.array(initial_indices), random_state = 42, n_cpu = 8, )
            fit_time_kmedoids= time.process_time() - start_fit_kmedoids
    
            centroids = np.array([np.array(x) for x in kmeans_instance.get_centers()])
            distances_to_centroids = cdist(centroids, X_train_fold)  # Compute pairwise distances
            closest_indices_to_centroids = np.argmin(distances_to_centroids, axis=1)  # Find the index of the closest point for each centroid
            
            medoids_indexes = kmedoids_instance.medoids

            centroids_indexes = closest_indices_to_centroids
            medoids_indexes = kmedoids_instance.medoids
            
            medoids = X_train_fold[medoids_indexes]
    
            predict_time = 0.0
            
            
            prototypes = {
                "kmeans": (centroids,centroids_indexes,fit_time_kmeans),
                "kmedoids": (medoids, medoids_indexes, fit_time_kmedoids)
            
            }
    
            for subset_key, prototypes in prototypes.items():
                factual_prototypes = prototypes[0]
                indexes_prototypes = prototypes[1]
                fit_time = prototypes[2]
                
                X_train_reduced = pairwise_distances(X_train_fold, factual_prototypes , metric = metric)
                X_test_reduced = pairwise_distances(X_test_fold, factual_prototypes , metric = metric)
                
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
                            "Additional Classifier" : f'{k}_{subset_key}_{clf_name}',
                            "Additional Classifier Fit Time (s)" : start_fit_extra,
                            "Additional Classifier Predict Time (s)" : predict_time_extra,
                            "n_prots_found" : len(indexes_prototypes),
                            "n_prots_used":len(indexes_prototypes),
                            **metrics,
                          #  **stump_counts
                        }
                    
                    results.append(result_entry)
                    
    
                    
                    if 'knn_' in clf_name:
                        try:
                            # Fit classifier
                            start_fit_extra = time.process_time()
                            clf.fit(X_train_fold[indexes_prototypes], y_train_fold[indexes_prototypes])
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
                            "Fold": fold_idx + 1,
                            "Fit Time (s)": fit_time,
                            "Predict Time (s)": predict_time,
                            "Additional Classifier": f'{k}_{subset_key}_{clf_name}_feature_space',
                            "Additional Classifier Fit Time (s)": fit_time_extra,
                            "Additional Classifier Predict Time (s)": predict_time_extra,
                            "n_prots_found" : len(indexes_prototypes),
                            "n_prots_used":len(indexes_prototypes),
                            **metrics,
                      #      **stump_counts
                        }
                        results.append(result_entry)
    
    
                            
                           
    
                    
    results_df = pd.DataFrame(results)
    
    
    # In[142]:
    
    
    results_df
    
    
    # In[146]:
    
    results_df.to_csv(f'datasets/KME_MED_RIGHT_testing/mstz_{dataset_name}_results.csv', index = False)
    
    
    grouped_results = results_df.groupby(
            ['distance_measure'	, 'random_state',	'Additional Classifier']
        ).agg(['mean', 'std'])
    
    
    # In[147]:
    
    
    # Flatten the multi-level columns
    grouped_results.columns = ['_'.join(col).strip() for col in grouped_results.columns.values]
    
    # Reset the index for a cleaner output
    grouped_results = grouped_results.reset_index()
    
    
    
    # In[18]:
    
    
    grouped_results.sort_values(by=['Weighted F1 Score_mean'], ascending=False)
    
    
    
    grouped_results.to_csv(f'datasets/KME_MED_RIGHT_testing/mstz_{dataset_name}_grouped.csv', index = False)
    
    
    
    
    
    
    # In[ ]:
    
    
    
    
    
    # In[ ]:
    
    
    
    
    
    # In[ ]:
    
    
    
    
    
    # In[ ]:
    
    
    
    
