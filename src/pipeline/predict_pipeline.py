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
            # Load model and preprocessor paths
            model_path = ModelTrainerConfig().trained_model_file_path
            preprocessor_path = DataTransformationConfig().preprocessor_obj_file_path

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # Transform input data
            transformed_data = preprocessor.transform(features)

            # Make prediction
            prediction = model.predict(transformed_data)

            return prediction
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 age_first_funding_year: float,
                 age_last_funding_year: float,
                 age_first_milestone_year: float,
                 age_last_milestone_year: float,
                 funding_rounds: int,
                 funding_total_usd: float,
                 milestones: int,
                 is_CA: int,
                 is_NY: int,
                 is_MA: int,
                 is_TX: int,
                 is_otherstate: int,
                 is_software: int,
                 is_web: int,
                 is_mobile: int,
                 is_enterprise: int,
                 is_advertising: int,
                 is_gamesvideo: int,
                 is_ecommerce: int,
                 is_biotech: int,
                 is_consulting: int,
                 is_othercategory: int,
                 has_VC: int,
                 has_angel: int,
                 has_roundA: int,
                 has_roundB: int,
                 has_roundC: int,
                 has_roundD: int,
                 avg_participants: float,
                 is_top500: int,
                 founded_at_month: int,
                 founded_at_day: int,
                 founded_at_year: int,
                 closed_at_month: int,
                 closed_at_day: int,
                 closed_at_year: int,
                 has_RoundABCD: int,
                 has_Investor: int,
                 has_Seed: int,
                 invalid_startup: int,
                 tier_relationships: int
                 ):
        self.age_first_funding_year = age_first_funding_year
        self.age_last_funding_year = age_last_funding_year
        self.age_first_milestone_year = age_first_milestone_year
        self.age_last_milestone_year = age_last_milestone_year
        self.funding_rounds = funding_rounds
        self.funding_total_usd = funding_total_usd
        self.milestones = milestones
        self.is_CA = is_CA
        self.is_NY = is_NY
        self.is_MA = is_MA
        self.is_TX = is_TX
        self.is_otherstate = is_otherstate
        self.is_software = is_soft
