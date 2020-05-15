import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Stacked_Model_Regressor:
    """A simple way to form a stacked model for regression problems.

    :param first_models: A list of models (e.g. from sklearn) going to be stacked.
    :param last_model: A model (e.g. from sklearn) going to perfom the final prediction.
    :rtype: A :class: 'Classifier <Classifier>'
    """

    def __init__(self, first_models, last_model):
        self.n_models = len(first_models)
        self.models = first_models
        self.final_model = last_model



    def fit(self, X, y, verbose = True):
        """Fit all the models with X and y.

        :param X: A pandas DataFrame or array.
        :param y: A pandas DataFrame or array.
        :param verbose: A boolean.
        :rtype: None
        """

        X_train, X_valid, y_train, y_valid = train_test_split(X, y)

        if verbose:
            print("Start training on {} observations and train the final model on {} observations...".format(X_train.shape[0], X_valid.shape[0]))

        count = 0

        predframe = pd.DataFrame()
        for model in self.models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            name = "y_pred_" + str(count)
            predframe[name] = y_pred
            count += 1

        self.final_model.fit(predframe, y_valid)

        if verbose:
            print("Finished.")



    def predict(self, X_test, proba=False):
        """Predict the label with the information from X_test.

        :param X: A pandas DataFrame or numpy array.
        :rtype: A numpy array.
        """

        count = 0

        predframe = pd.DataFrame()
        for model in self.models:
            y_pred = model.predict(X_test)
            name = "y_pred_" + str(count)
            predframe[name] = y_pred
            count += 1

        y_pred_final = self.final_model.predict(predframe)
        return y_pred_final



    def evaluate(self, X_test, y_test, metric, all=True, digits=4):
        """Score the performance of the model and of all the models if asked, with a custom metric and with digits of precision.

        :param X_test: A pandas DataFrame or numpy array.
        :param y_test: A pandas DataFrame or numpy array.
        :param metric: A function.
        :param all: A boolean.
        :param digits: An integer.
        :rtype: None.
        """

        count = 0

        predframe = pd.DataFrame()
        for model in self.models:
            y_pred = model.predict(X_test)
            name = "y_pred_" + str(count)
            predframe[name] = y_pred
            count += 1

        if all:
            for i in range(self.n_models):
                performance = round(np.sqrt(metric(y_test, predframe["y_pred_" + str(i)])), digits)
                print("Model {}, performance : {}".format(i, performance))

        y_pred_final = self.final_model.predict(predframe)
        performance = round(np.sqrt(metric(y_test, y_pred_final)), digits)
        print("Stacked model performance : {}".format(performance))













class Stacked_Model_Classifier:
    """A simple way to form a stacked model for classification problems.

    :param first_models: A list of models (e.g. from sklearn) going to be stacked.
    :param last_model: A model (e.g. from sklearn) going to perfom the final prediction.
    :rtype: A :class: 'Classifier <Classifier>'
    """

    def __init__(self, first_models, last_model):
        self.n_models = len(first_models)
        self.models = first_models
        self.final_model = last_model



    def fit(self, X, y, verbose=True, test_size=0.25):
        """Fit all the models with X and y.

        :param X: A pandas DataFrame or array.
        :param y: A pandas DataFrame or array.
        :param verbose: A boolean.
        :rtype: None
        """

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=test_size)

        if verbose:
            print("Start training on {} observations and train the final model on {} observations...".format(X_train.shape[0], X_valid.shape[0]))

        count = 0

        predframe = pd.DataFrame()
        for model in self.models:
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_valid)
            name = "y_pred_" + str(count)
            predframe[name] = y_pred[:, 0]
            count += 1

        self.final_model.fit(predframe, y_valid)

        if verbose:
            print("Finished.")



    def predict(self, X_test):
        """Predict the label with the information from X_test.

        :param X: A pandas DataFrame or numpy array.
        :rtype: A numpy array.
        """

        count = 0

        predframe = pd.DataFrame()
        for model in self.models:
            y_pred = model.predict_proba(X_test)
            name = "y_pred_" + str(count)
            predframe[name] = y_pred[:, 0]
            count += 1

        y_pred_final = self.final_model.predict(predframe)
        return y_pred_final



    def evaluate(self, X_test, y_test, metric, all=True, digits=4):
        """Score the performance of the model and of all the models if asked, with a custom metric and with digits of precision.

        :param X_test: A pandas DataFrame or numpy array.
        :param y_test: A pandas DataFrame or numpy array.
        :param metric: A function.
        :param all: A boolean.
        :param digits: An integer.
        :rtype: None.
        """
        
        count = 0

        if all:
            for model in self.models:
                y_pred = model.predict(X_test)
                performance = round(np.sqrt(metric(y_test, y_pred)), digits)
                print("Model {}, performance : {}".format(count, performance))
                count += 1

        y_pred = self.predict(X_test)
        performance = round(np.sqrt(metric(y_test, y_pred)), digits)
        print("Stacked model performance : {}".format(performance))
