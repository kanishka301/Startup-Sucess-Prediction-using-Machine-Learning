import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(obj, file_path):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict = None, is_binary=True):
   
    try:
        evaluation_report = {}

        for name, model in models.items():
            print(f"\nEvaluating: {name}")

            # GridSearchCV only if parameters provided for the model
            if param and name in param:
                grid = GridSearchCV(estimator=model, param_grid=param[name], cv=3, n_jobs=-1, scoring='accuracy')
                grid.fit(X_train, y_train)
                model = grid.best_estimator_

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            try:
                if is_binary:
                    y_test_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                else:
                    y_test_roc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
            except:
                y_test_roc = 0.0

            print(f"  Test Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
            print(f"  Test F1 Score : {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
            print(f"  ROC AUC Score : {y_test_roc:.4f}")

            evaluation_report[name] = accuracy_score(y_test, y_test_pred)

        return evaluation_report

    except Exception as e:
        raise CustomException(e, sys)
