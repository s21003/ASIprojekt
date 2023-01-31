"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.18.4
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, roc_auc_score
# from sklearn import metrics

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import MinMaxScaler

import joblib
import logging


def prepare_data_for_modeling(df):
    pd.options.mode.chained_assignment = None
    data = df.columns[1:-1]
    x = df[data]
    y = df["price"]
    numeric_data = x.select_dtypes(include=[np.number]).columns
    categorical_data = x.select_dtypes(exclude=[np.number]).columns
    scaler = MinMaxScaler()
    x[numeric_data] = scaler.fit_transform(x[numeric_data])
    x = pd.get_dummies(x, columns=categorical_data)
    y = y.astype(int)
    data_prepared = pd.concat([x, y], axis=1)
    return data_prepared


def split_data(df):
    data = df.columns[:-1]
    x = df[data]
    y = df["price"]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
    return xtrain, xtest, ytrain, ytest


def train_model(xtrain, ytrain):
    pd.options.mode.chained_assignment = None
    # model = LogisticRegression(solver='lbfgs', max_iter=200)
    model = DecisionTreeRegressor()
    model.fit(xtrain, ytrain)
    return model


# def evaluate_model(model, xtest, ytest):
#     labels = ytest.unique()
#     y_pred = model.predict(xtest)
#     y_probas = model.predict_proba(xtest)[:, 1]

#     accuracy = accuracy_score(ytest, y_pred)
#     roc_auc = roc_auc_score(ytest, y_probas)
#     print("ROC AUC: %.3f" % roc_auc)
#     print("Accuracy: %.3f" % accuracy)

#     logger = logging.getLogger(__name__)
#     logger.info("Model has an accuracy of %.3f on test data.", accuracy)
#     logger.info("Model has an ROC AUC of %.3f on test data.", roc_auc)


def evaluate_model(model, xtest, ytest):
    predictions = model.predict(xtest)

    logger = logging.getLogger(__name__)
    logger.info(
        "Model ma dokładność: %.3f na test setcie.", model.score(xtest, predictions)
    )
