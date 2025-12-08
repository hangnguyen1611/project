import logging
import pandas as pd
import numpy as np

from .model_config import MODEL_REGISTRY
from ..logs import logger
from .grid_tuner import GridTuner
from .model_trainer import ModelTrainer
from .optuna_tuner import OptunaTuner
from .evaluator import EvaluateModel

logger.setup_logger()
logger = logging.getLogger(__name__)

class BestModelSelector:
    """
    Lớp này dùng để chọn mô hình tốt nhất.
    """
    def __init__(self, X_train, X_test, y_train, y_test, method="optuna", scoring=None, random_seed=42):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.method = method
        self.scoring = scoring
        self.random_seed = random_seed

    def select(self):
        """
        Chọn mô hình tốt nhất.

        Trả về Dictionary gồm:
            - best_model_name: tên model tốt nhất.
            - best_model: model tốt nhất đã tối ưu tham số.
            - all_metrics: chỉ số đánh giá cho tất cả các mô hình.
        """
        logger.info("Chọn mô hình tốt nhất...")

        n_classes = self.y_train.nunique()
        scoring = self.scoring or ("roc_auc" if n_classes == 2 else "f1")

        results = []
        best_score = -np.inf
        best_model = None
        best_name = None

        for name, cfg in MODEL_REGISTRY.items():
            logger.info(f"Training {name}...")

            trainer = ModelTrainer(model_name=name, seed=self.random_seed)
            model = trainer.fit(self.X_train, self.y_train, self.X_test, self.y_test)

            metrics = EvaluateModel(
                self.X_test,
                self.y_test,
                model
            ).evaluate()

            metrics["model_name"] = name
            results.append(metrics)

            # Tính score
            if scoring == "roc_auc":
                if "roc_auc" in metrics.columns:
                    score = metrics["roc_auc"].iloc[0]
                elif "roc_auc_ovr" in metrics.columns:
                    score = metrics["roc_auc_ovr"].iloc[0]
                else:
                    score = metrics["f1"].iloc[0]    # fallback
            else:
                score = metrics["f1"].iloc[0]

            logger.info(f"{name} {scoring}: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

        logger.info(f"Mô hình được chọn: {best_name}, score = {best_score:.4f}")
        logger.info(f"Tối ưu tham số cho {best_name}...")

        if self.method == "grid":
            tuner = GridTuner(self.X_train, self.y_train, seed=self.random_seed)
            best_model = tuner.optimize(best_name)
        else:
            tuner = OptunaTuner(self.X_train, self.y_train, seed=self.random_seed)
            best_model = tuner.optimize(best_name, scoring=scoring)

        all_metrics = pd.concat(results, ignore_index=True)
        cols = ["model_name"] + [c for c in all_metrics.columns if c != "model_name"]
        all_metrics = all_metrics[cols]

        logger.info(f"Best model: {best_name} with optimized params.")
        return {
            "best_model_name": best_name,
            "best_model": best_model,
            "all_metrics": all_metrics.sort_values(scoring, ascending=False)
        }
