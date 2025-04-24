# PivotTree

This repository contains the code for PivotTree and RandomPivotTree, a case-based hierarchical decision tree based model providing an interpretable decision making tool for classification.


```validation``` and ```testing``` and respectively include code to run testing and validation experiments. Ensure to run the ```.py``` validation and testing file in the same folder containing ```RuleTree``` folder for proper running or modify the import in the code above to perform stumps and utils import


```BoundaryVisualization.ipynb``` shows code with deptiction of differnet boundary of univariate, oblique and proximity based splits for different versions of ```PivotTree```

```TimeSeriesVisualization.ipynb``` shows code of training univariate and proximity based splits for ```gun``` dataset for the time-sereis qualitative example reported. Similar code is used for ```cifar10``` and ```oral``` datasets as examples depicted in the relative paper


![img_N_cifar10_univar (1)](https://github.com/user-attachments/assets/b79bf60c-5f43-460b-9039-b5e55bb121c5)
![](path_to_image)
*Example of Univariate Pivot Tree with maxdepth = 4 on ```cifar10```. Only partial structure shown for visualization purposes.*




![img_N_cifar10_prox (1)](https://github.com/user-attachments/assets/b9570008-713a-4674-9bfc-218d9011e7f6)
![](path_to_image)
*Example of Proximity Pivot Tree with maxdepth = 4 on ```cifar10```. Only partial structure shown for visualization purposes.*


## Training trees
```PivotTree``` follows the classic sklearn `fit`/`predict` interface.  

```python
from sklearn.datasets import load_iris

from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from RuleTree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
from RuleTree.stumps.instance_stumps import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Pivot Tree Classifier

pt_stump = pt_stump_call()

clf = RuleTreeClassifier(distance_measure='euclidean',
       base_stumps = [pt_stump],
       stump_selection = 'best',
       random_state = 0,  max_depth = 3)

clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

```

The kind of stumps in ```instance_stumps``` determines the kind of split for pivots used. Other options can be applied based on oblique or proximity splits such asw 


```python
proximity_stump = multi_pt_stump_call()

clf = RuleTreeClassifier(distance_measure='euclidean',
       base_stumps = [proximity_stump],
       stump_selection = 'best',
       random_state = 0,  max_depth = 3)
```

or 

```python
obl_pt_stump = obl_pt_stump_call()

clf = RuleTreeClassifier(distance_measure='euclidean',
       base_stumps = [obl_pt_stump],
       stump_selection = 'best',
       random_state = 0,  max_depth = 3)
```


## Training forests
For ```RandomPivotForest``` a similar apporach can be used for training and prediction:

```python
from sklearn.datasets import load_iris

from RuleTree.tree.RuleTreeClassifier import RuleTreeClassifier
from RuleTree.stumps.classification.PivotTreeStumpClassifier import PivotTreeStumpClassifier
from RuleTree.ensemble.RuleForestClassifier import RuleForestClassifier
from RuleTree.stumps.instance_stumps import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Pivot Forest Classifier

pt_stump = pt_stump_call()

clf = RuleForestClassifier(distance_measure='euclidean', 
                            base_stumps = [pt_stump], 
                            stump_selection = 'best', 
                            random_state = 42, 
                            max_depth = 2,
                            n_estimators = 100,
                            max_samples=0.1,
                            max_features=0.1,
                            bootstrap = False,
                            bootstrap_features = False,
                            )

clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

```
