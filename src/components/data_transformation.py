import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self, input_features: list):
        try:
            passthrough_transformer = ColumnTransformer([
                ("pass_through", FunctionTransformer(validate=False), input_features)
            ])
            logging.info("Returning passthrough transformer (no scaling).")
            return passthrough_transformer

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Reading training and testing data.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "status"
            input_features = [col for col in train_df.columns if col != target_column]

            preprocessor = self.get_data_transformer_object(input_features)

            X_train = train_df[input_features]
            y_train = train_df[target_column]

            X_test = test_df[input_features]
            y_test = test_df[target_column]

            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            logging.info("Saving preprocessor object.")
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            train_arr = np.c_[X_train_processed, np.array(y_train)]
            test_arr = np.c_[X_test_processed, np.array(y_test)]

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
