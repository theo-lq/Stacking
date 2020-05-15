import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold





class StackedClassifier:
    """A simple way to form a stacked model for classification problems.

    :param models: A list of models (e.g. from sklearn) going to be stacked.
    :param meta_model: A model (e.g. from sklearn) going to perfom the final prediction.
    :rtype: A :class: 'Classifier <Classifier>'
    """

    
    def __init__(self, models, meta_model, interest_class=1):
        self.n_models = len(models)
        self.models = models
        self.meta_model = meta_model
        self.interest_class = interest_class



    def fit(self, X, y, cv=10):
        """Fit all the models with X and y with cv folds.
        
        :param X: A pandas DataFrame or array.
        :param y: A pandas DataFrame or array.
        :param cv: A integer, positive.
        :rtype: None
        """
        
        indexes_generator = StratifiedKFold(n_splits=cv, shuffle=True).split(X, y)
        indexes = [(train_index, test_index) for train_index, test_index in indexes_generator]
        
        count = 0
        
        predframe = pd.DataFrame()
        for model in self.models:
            y_pred_model = []
            for train_index, test_index in indexes:
                X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_test)[:, self.interest_class]
                y_pred_model.extend(y_pred)
            
            name = "y_pred_" + str(count)
            predframe[name] = y_pred_model
            count += 1
        
        y_for_meta_model = []
        for _, test_index in indexes:
            y_for_meta_model.extend(y.iloc[test_index])
            
        self.meta_model.fit(predframe, y_for_meta_model)
    
    
    
    def predict(self, X):
        """Predict the label with the information from X.
        
        :param X: A pandas DataFrame or numpy array.
        :rtype: A numpy array.
        """
        
        count = 0
        
        predframe = pd.DataFrame()
        for model in self.models:
            y_pred = model.predict_proba(X)[:, self.interest_class]
            name = "y_pred_" + str(count)
            predframe[name] = y_pred
            count += 1
        
        y_pred_final = self.meta_model.predict(predframe)
        return y_pred_final



    def evaluate(self, X, y, metric, all=True):
        """Score the performance of the model and of all the models if asked.
        
        :param X: A pandas DataFrame or numpy array.
        :param y: A pandas DataFrame or numpy array.
        :param metric: A function.
        :param all: A boolean.
        :rtype: None.
        """
        
        count = 0
        
        predframe = pd.DataFrame()
        for model in self.models:
            y_pred = model.predict_proba(X)[:, self.interest_class]
            name = "y_pred_" + str(count)
            predframe[name] = y_pred
            
            if all:
                y_pred = model.predict(X)
                performance = metric(y, y_pred)
                print("Model %1.0f, performance : %0.4f" % (count, performance))
            
            count += 1
        
        
        y_pred_final = self.meta_model.predict(predframe)
        performance = metric(y, y_pred_final)
        print("Stacked model performance : %0.4f" % performance)