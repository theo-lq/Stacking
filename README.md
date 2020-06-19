# Stacking
 Implementation of the Stacking method for both regression and classification.

To use it, define for each layer a list of models you want to use. Then create a list to put all these layers in order. For example, if we consider you already have your $X$ and $y$ matrix/vector :

 ```python
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y)
 
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import NuSVR
from sklearn.linear_model import LinearRegression
from sklearn.lightgbm import LGBMRegressor

layer_1 = [RandomForestRegressor(n_estimators=300), NuSVR(gamma='scale'), LinearRegression()] 
layer_2 = [RandomForestRegressor(n_estimators=200), NuSVR(gamma='scale'), LinearRegression()]
layer_3 = [RandomForestRegressor(n_estimators=100), NuSVR(gamma='scale'), LinearRegression()]

models = [layer_1, layer_2, layer_3]
final_model = LGBMRegressor(n_estimators=100)

model = StackedRegressor(models, final_model)

from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred): return mean_squared_error(y_true=y_true, y_pred=y_pred) ** 0.5

model.evaluate(X_test, y_test, metric=rmse)
```

Then you can use all the methods defined.
