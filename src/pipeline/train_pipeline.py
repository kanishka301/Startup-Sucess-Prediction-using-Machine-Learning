from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # Step 1: Ingest Data
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Step 2: Transform Data
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

    # Step 3: Train and Save Model
    trainer = ModelTrainer()
    model_accuracy = trainer.initiate_model_trainer(train_arr, test_arr)

print(f"Model training completed. Test Accuracy: {model_accuracy}")


