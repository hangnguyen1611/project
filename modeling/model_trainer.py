import logging
from typing import Optional
import inspect
import os, contextlib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from ..logs import logger

logger.setup_logger()
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Lớp này dùng để huấn luyện mô hình:
        - Huấn luyện nhanh bằng model_name (baseline).
        - Huấn luyện model đã được tune (Grid / Optuna).
    """

    def __init__(self, model=None, model_name = None, seed = 42):
        if model is None and model_name is None:
            raise ValueError("Phải truyền model hoặc model_name")

        self.random_seed = seed
        self.model = model or self._build_baseline_model(model_name)

    def _build_baseline_model(self, name):
        """
        Tạo baseline model.

        Tham số đầu vào:
            - name: tên mô hình muốn train.

        Trả về mô hình đã build baseline.
        """
        logging.info(f"Building baseline model: {name}")

        model_map = {
            "logistic": LogisticRegression(
                max_iter=1000,
                random_state=self.random_seed
            ),
            "randomforest": RandomForestClassifier(
                random_state=self.random_seed,
                n_jobs=-1
            ),
            "xgboost": XGBClassifier(
                random_state=self.random_seed,
                verbosity=0
            ),
            "lightgbm": LGBMClassifier(
                random_state=self.random_seed,
                verbosity=-1
            )
        }

        if name not in model_map:
            raise ValueError(f"model_name phải thuộc {list(model_map.keys())}")

        return model_map[name]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit model.

        Tham số đầu vào:
            - X_train, y_train: các tập huấn luyện.
            - X_val, y_val: các tập kiểm tra.
        
        Trả về mô hình đã fit.
        """
        model_name = self.model.__class__.__name__
        logger.info(f"Training {model_name}")

        fit_params = {}

        # Nếu model hỗ trợ eval_set và có dữ liệu validation
        if X_val is not None and y_val is not None:
            if "eval_set" in inspect.signature(self.model.fit).parameters:
                fit_params["eval_set"] = [(X_val, y_val)]
                if model_name == "xgboost":
                    fit_params["verbose"] = False
        
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            self.model.fit(X_train, y_train, **fit_params)

        logger.info(f"{model_name} trained successfully")
        return self.model
