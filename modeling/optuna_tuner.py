import numpy as np
import optuna
import logging
import os, contextlib

from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score
from .model_config import MODEL_REGISTRY
from logs import logger

logger.setup_logger()
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

class OptunaTuner:
    """
    Tối ưu tham số bằng Optuna.
    """
    def __init__(self, X, y, seed=42):
        self.X = X
        self.y = y
        self.seed = seed
        self.n_classes = len(np.unique(y))

    def _evaluate(self, model, X_val, y_val, scoring, use_proba=True):
        """
        Đánh giá cho trial.

        Tham số đầu vào:
            - model: mô hình đang tối ưu.
            - X_val, y_val: tập giá trị để kiểm tra.
            - scoring: chỉ số đánh giá.
            - use_proba: có sử dụng proba hay không (tính roc_auc).

        Trả về điểm số cho trial.
        """
        if scoring == "roc_auc":
            if not use_proba:
                raise ValueError("Model does not support predict_proba")

            proba = model.predict_proba(X_val)
            return roc_auc_score(
                y_val,
                proba[:, 1] if self.n_classes == 2 else proba,
                multi_class="ovr" if self.n_classes > 2 else None,
                average="macro" if self.n_classes > 2 else None
            )

        return f1_score(
            y_val,
            model.predict(X_val),
            average="binary" if self.n_classes == 2 else "macro"
        )

    def optimize(self, model_config, n_trials=30, cv=3, scoring=None):
        """
        Hàm tối ưu tham số chính.

        Tham số đầu vào:
            - model_config: tên model hoặc model đã khởi tạo.
            - n_trials: số trial.
            - cv: số fold cho cross validation.
            - scoring: chỉ số đánh giá.

        Trả về: mô hình đã fit với tham số tối ưu.
        """
        scoring = scoring or ("roc_auc" if self.n_classes == 2 else "f1")

        if isinstance(model_config, str):
            model_config = MODEL_REGISTRY[model_config]

        logger.info("────────────────────────────────────────────────────────────────────────────────────────────")
        logger.info(f"[Optuna] Bắt đầu tối ưu tham số cho mô hình: {model_config.name}")
        logger.info(f"[Optuna] Scoring: {scoring}")
        logger.info(f"[Optuna] Số trial: {n_trials}, CV folds: {cv}")
        logger.info(f"[Optuna] Số lớp: {self.n_classes}")

        def objective(trial):
            params = model_config.param_space(trial, self.n_classes, self.y, self.seed)

            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.seed)
            scores = []

            for fold, (tr_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
                model = model_config.model_cls(**params)

                eval_set = [(self.X.iloc[val_idx], self.y.iloc[val_idx])]

                if hasattr(model, "set_params") and "verbosity" in model.get_params():
                    model.set_params(verbosity=0)

                # Chỉ truyền eval_set nếu model hỗ trợ
                fit_params = {}
                if hasattr(model, "fit") and "eval_set" in model.fit.__code__.co_varnames:
                    fit_params["eval_set"] = eval_set
                    if isinstance(model, XGBClassifier):
                        fit_params["verbose"] = False  

                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    model.fit(self.X.iloc[tr_idx], self.y.iloc[tr_idx], **fit_params)

                score = self._evaluate(
                    model,
                    self.X.iloc[val_idx],
                    self.y.iloc[val_idx],
                    scoring,
                    model_config.use_predict_proba
                )

                trial.report(score, fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                scores.append(score)
            mean_score = float(np.mean(scores))
            logger.info(f"[Optuna] Trial {trial.number} params: {params} - mean_score: {mean_score}")

            return mean_score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )

        study.optimize(objective, n_trials=n_trials)

        # Build best model
        best_params = study.best_trial.params
        best_params.update({"random_state": self.seed})

        logger.info(f"[Optuna] Best Params: {best_params}")
        logger.info(f"[Optuna] Best CV Score: {study.best_value:.4f}")
        logger.info("────────────────────────────────────────────────────────────────────────────────────────────")

        best_model = model_config.model_cls(**best_params)
        best_model.fit(self.X, self.y)

        return best_model