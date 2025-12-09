from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Base ModelConfig
@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_cls: Type[BaseEstimator]

    param_space: Optional[Callable] = None   # Optuna
    param_grid: Optional[Dict] = None        # GridSearch

    use_predict_proba: bool = True

# Logistic Regression
def logreg_param_space(trial, n_classes, y, seed):
    return {
        "C": trial.suggest_float("C", 1e-3, 100, log=True),
        "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
        "max_iter": 2000,
        "random_state": seed
    }

LOGISTIC_CONFIG = ModelConfig(
    name="logistic",
    model_cls=LogisticRegression,
    param_space=logreg_param_space,
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # liblinear hỗ trợ l1/l2
    }
)

# Random Forest
def rf_param_space(trial, n_classes, y, seed):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": seed,
        "n_jobs": 1
    }

RANDOM_FOREST_CONFIG = ModelConfig(
    name="randomforest",
    model_cls=RandomForestClassifier,
    param_space=rf_param_space,
    param_grid={
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
)

# XGBoost
def xgb_param_space(trial, n_classes, y, seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": 0
    }

    if n_classes == 2:
        params.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss"
        })
    else:
        params.update({
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": n_classes
        })

    return params

XGBOOST_CONFIG = ModelConfig(
    name="xgboost",
    model_cls=XGBClassifier,
    param_space=xgb_param_space,
    param_grid={
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0]
    }
)

# LightGBM
def lgbm_param_space(trial, n_classes, y, seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 5, 7, 9]),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": -1
    }

    if n_classes == 2:
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        params.update({
            "objective": "binary",
            "scale_pos_weight": neg / pos if pos > 0 else 1.0
        })
    else:
        params.update({
            "objective": "multiclass",
            "num_class": n_classes
        })

    return params

LIGHTGBM_CONFIG = ModelConfig(
    name="lightgbm",
    model_cls=LGBMClassifier,
    param_space=lgbm_param_space,
    param_grid={
        'num_leaves': [31, 63, 127],
        'max_depth': [-1, 10],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 1.0]
    }
)

# Registry
MODEL_REGISTRY = {
    "logistic": LOGISTIC_CONFIG,
    "randomforest": RANDOM_FOREST_CONFIG,
    "xgboost": XGBOOST_CONFIG,
    "lightgbm": LIGHTGBM_CONFIG
}