import numpy as np
import pandas as pd
from sklearn.model_selection import KFold





class StackedRegressor:

    def __init__(self, models, meta_model):
        self.levels = len(models)
        self.models = models
        self.meta_model = meta_model



    def fit(self, X_train, y_train, cv=10):
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
                    performance = round(metric(y, y_pred), 4)
                    print("Level {} - Model {} : {}".format(level, model.__class__.__name__, performance))

                count += 1

            X = predframe


        y_pred_final = self.meta_model.predict(predframe)
        performance = metric(y, y_pred_final)
        print("Stacked model performance : %0.4f" % performance)
