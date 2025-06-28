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

            # Load model and preprocessor objects
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # Apply transformation (no scaling, just passthrough)
            transformed_data = preprocessor.transform(features)

            # Predict
            prediction = model.predict(transformed_data)
            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        age_first_funding_year: float,
        age_last_funding_year: float,
        age_first_milestone_year: float,
        age_last_milestone_year: float,
        funding_rounds: float,
        funding_total_usd: float,
        milestones: float,
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
        tier_relationships: float,
    ):
        self.data = {
            "age_first_funding_year": [age_first_funding_year],
            "age_last_funding_year": [age_last_funding_year],
            "age_first_milestone_year": [age_first_milestone_year],
            "age_last_milestone_year": [age_last_milestone_year],
            "funding_rounds": [funding_rounds],
            "funding_total_usd": [funding_total_usd],
            "milestones": [milestones],
            "is_CA": [is_CA],
            "is_NY": [is_NY],
            "is_MA": [is_MA],
            "is_TX": [is_TX],
            "is_otherstate": [is_otherstate],
            "is_software": [is_software],
            "is_web": [is_web],
            "is_mobile": [is_mobile],
            "is_enterprise": [is_enterprise],
            "is_advertising": [is_advertising],
            "is_gamesvideo": [is_gamesvideo],
            "is_ecommerce": [is_ecommerce],
            "is_biotech": [is_biotech],
            "is_consulting": [is_consulting],
            "is_othercategory": [is_othercategory],
            "has_VC": [has_VC],
            "has_angel": [has_angel],
            "has_roundA": [has_roundA],
            "has_roundB": [has_roundB],
            "has_roundC": [has_roundC],
            "has_roundD": [has_roundD],
            "avg_participants": [avg_participants],
            "is_top500": [is_top500],
            "founded_at_month": [founded_at_month],
            "founded_at_day": [founded_at_day],
            "founded_at_year": [founded_at_year],
            "closed_at_month": [closed_at_month],
            "closed_at_day": [closed_at_day],
            "closed_at_year": [closed_at_year],
            "has_RoundABCD": [has_RoundABCD],
            "has_Investor": [has_Investor],
            "has_Seed": [has_Seed],
            "invalid_startup": [invalid_startup],
            "tier_relationships": [tier_relationships]
        }

    def get_data_as_df(self):
        try:
            return pd.DataFrame(self.data)
        except Exception as e:
            raise CustomException(e, sys)
