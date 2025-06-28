import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing arrays.")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "CatBoost": CatBoostClassifier(verbose=0),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }

            # Only tune XGBoost
            params = {
                "XGBoost": {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5]
                }
            }

            logging.info("Evaluating models...")
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
                is_binary=True
            )

            # Get the best model based on test accuracy
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_score = model_report[best_model_name]

            if best_score < 0.6:
                raise CustomException("No model met the minimum accuracy threshold.")

            logging.info(f"Best model found: {best_model_name} with accuracy: {best_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Final test accuracy
            y_pred = best_model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)

            return f"Best Model: {best_model_name}, Accuracy: {round(test_acc, 4)}"

        except Exception as e:
            raise CustomException(e, sys)
