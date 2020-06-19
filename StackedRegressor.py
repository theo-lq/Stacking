import numpy as np
import pandas as pd
from sklearn.model_selection import KFold





class StackedRegressor:
    """A simple way to form a stacked model for regression problems.

    :param models: A list of models (e.g. from sklearn) going to be stacked.
    :param meta_model: A model (e.g. from sklearn) going to perfom the final prediction.
    :rtype: A :class: 'Classifier <Classifier>'
    """
    
    
    def __init__(self, models, meta_model):
        self.levels = len(models)
        self.models = models
        self.meta_model = meta_model



    def fit(self, X_train, y_train, cv=5):
        """Fit all the models with X_train and y_train with cv folds.
        
        :param X_train: A pandas DataFrame or array.
        :param y_train: A pandas DataFrame or array.
        :param cv: A integer, positive.
        :rtype: None
        """
        
        X = X_train.copy()
        y = y_train.copy()

        for level in range(self.levels):
            count = 0
            predframe = pd.DataFrame()

            indexes_generator = KFold(n_splits=cv, shuffle=True).split(X)
            indexes = [(train_index, test_index) for train_index, test_index in indexes_generator]

            for model in self.models[level]:
                y_pred_model = []
                for train_index, test_index in indexes:
                    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_model.extend(y_pred)

                name = "y_pred_" + str(count)
                predframe[name] = y_pred_model
                count += 1

            X = predframe
            y = []
            for _, test_index in indexes:
                y.extend(y.iloc[test_index])

        self.meta_model.fit(X, y)



    def predict(self, X_test):
        """Predict the label with the information from X_test.
        
        :param X_test: A pandas DataFrame or numpy array.
        :rtype: A numpy array.
        """
        
        X = X_test.copy()

        for level in range(self.levels):
            count = 0
            predframe = pd.DataFrame()

            for model in self.models[level]:
                y_pred = model.predict(X)
                name = "y_pred_" + str(count)
                predframe[name] = y_pred
                count += 1

            X = predframe

        y_pred_final = self.meta_model.predict(predframe)
        return y_pred_final



    def evaluate(self, X_test, y_test, metric, all=True):
        """Score the performance of the model and of all the models if asked.
        
        :param X_test: A pandas DataFrame or numpy array.
        :param y_test: A pandas DataFrame or numpy array.
        :param metric: A function.
        :param all: A boolean.
        :rtype: None.
        """
        X = X_test.copy()
        y = y_test.copy()

        for level in range(self.levels):
            count = 0
            predframe = pd.DataFrame()
            for model in self.models[level]:
                y_pred = model.predict(X)
                name = "y_pred_" + str(count)
                predframe[name] = y_pred

                if all:
                    performance = metric(y, y_pred)
                    print("Level %d - Model %s : %0.4f".format(level, model.__class__.__name__, performance))

                count += 1

            X = predframe


        y_pred_final = self.meta_model.predict(predframe)
        performance = metric(y, y_pred_final)
        print("Stacked model performance : %0.4f" % performance)
