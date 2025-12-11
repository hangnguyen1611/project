from .pipeline_model import ModelTrainPipeline
import joblib
import logging
from logs import logger

logger.setup_logger()
logger = logging.getLogger(__name__)

def run_training(processed_csv: str, target: str, file_name: str):
    # Load preprocessing artifacts
    encoders = joblib.load("artifacts/encoders.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")

    # Train model
    pipe = (
        ModelTrainPipeline(random_seed=42, scaler=scaler, encoders=encoders)
        .load_data(processed_csv, target)
        .split_data()
    )
        
    pipe.select_best_model()
    metrics = pipe.evaluate(encoders=encoders)
    print("Evaluation metrics:", metrics)

    # Save best model
    pipe.save(output_path_str=file_name)
    print(f"Model tốt nhất được lưu tại artifacts/{file_name}")
