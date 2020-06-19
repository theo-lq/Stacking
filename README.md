# Stacking
 Implementation of the Stacking method for both regression and classification.

To use it, define for each layer a list of models you want to use. Then create a list to put all these layers in order. For example :

layer_1 = [RandomForestRegressor(n_estimators=100), NuSVR(gamma='scale'), ...] 
layer_2 = [...]
layer_3 = [...]

models = [layer_1, layer_2, layer_3]
final_model = LGBMRegressor(n_estimators=100)

model = StackedRegressor(models, final_model)

Then you can use all the methods defined.
