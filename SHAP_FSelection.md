```python
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split

iris_data = load_iris()

X, y = iris_data.data, iris_data.target
feature_names = np.array(iris_data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```


```python
feature_names
```




    array(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
           'petal width (cm)'], dtype='<U17')




```python
import catboost as cb

model = cb.CatBoostClassifier(verbose=False)
model.fit(X_train, y_train)
```




    <catboost.core.CatBoostClassifier at 0x7faa479cfe80>




```python
from shap_selection import feature_selection

# please, use agnostic = True to use with any model...
# agnostic = False will only work with tree-based models
feature_order = feature_selection.shap_select(model, X_train, X_test, feature_names, agnostic=False)
```


```python
feature_order
```




    (array(['petal width (cm)', 'petal length (cm)', 'sepal length (cm)',
            'sepal width (cm)'], dtype='<U17'),
     array([3.6433512 , 3.36917314, 1.44244029, 1.37962501]))


