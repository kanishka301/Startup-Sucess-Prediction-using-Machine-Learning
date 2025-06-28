import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
           
            model_path = ModelTrainerConfig().trained_model_file_path
            preprocessor_path = DataTransformationConfig().preprocessor_obj_file_path

            
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

         
            transformed_data = preprocessor.transform(features)

            # Predict
            prediction = model.predict(transformed_data)
            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 closed_at_month: int,
                 closed_at_year: int,
                 is_ecommerce: int,
                 age_last_milestone_year: float,
                 founded_at_month: int,
                 age_first_milestone_year: float,
                 is_CA: int,
                 age_last_funding_year: float,
                 is_MA: int,
                 tier_relationships: float):
        
        self.closed_at_month = closed_at_month
        self.closed_at_year = closed_at_year
        self.is_ecommerce = is_ecommerce
        self.age_last_milestone_year = age_last_milestone_year
        self.founded_at_month = founded_at_month
        self.age_first_milestone_year = age_first_milestone_year
        self.is_CA = is_CA
        self.age_last_funding_year = age_last_funding_year
        self.is_MA = is_MA
        self.tier_relationships = tier_relationships

    def get_data_as_df(self):
        try:
            data = {
                "closed_at_month": [self.closed_at_month],
                "closed_at_year": [self.closed_at_year],
                "is_ecommerce": [self.is_ecommerce],
                "age_last_milestone_year": [self.age_last_milestone_year],
                "founded_at_month": [self.founded_at_month],
                "age_first_milestone_year": [self.age_first_milestone_year],
                "is_CA": [self.is_CA],
                "age_last_funding_year": [self.age_last_funding_year],
                "is_MA": [self.is_MA],
                "tier_relationships": [self.tier_relationships]
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
