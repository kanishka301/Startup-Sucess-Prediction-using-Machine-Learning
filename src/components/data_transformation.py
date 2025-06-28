import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self, input_features: list):
        try:
            logging.info("Creating standard scaling pipeline for numeric features.")

            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, input_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Reading training and testing data.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "status"
            input_features = [col for col in train_df.columns if col != target_column]

            logging.info("Obtaining preprocessor object.")
            preprocessor = self.get_data_transformer_object(input_features)

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Fitting and transforming training data.")
            X_train_scaled = preprocessor.fit_transform(X_train)
            logging.info("Transforming test data.")
            X_test_scaled = preprocessor.transform(X_test)

            logging.info("Saving preprocessor object.")
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
