import argparse
import logging
from logs import logger

logger.setup_logger()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project ML pipeline runner")
    parser.add_argument(
        "--mode", 
        choices=["preprocess", "train"], 
        required=True,
        help="Mode to run"
    )
    parser.add_argument("--data", type=str, required=True, help="Input CSV path")
    parser.add_argument("--target", type=str, default="mental_health_risk", help="Target column")
    parser.add_argument("--file_name", type=str, default="best_model.pkl", help="File name of the best model")
    
    args = parser.parse_args()

    if args.mode == "preprocess":
        from preprocessing.runner_preprocess import run_preprocessing
        run_preprocessing(args.data)
    elif args.mode == "train":
        from modeling.runner_train import run_training
        run_training(args.data, args.target, args.file_name)