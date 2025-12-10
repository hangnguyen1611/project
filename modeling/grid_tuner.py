import numpy as np
import logging
import os, contextlib

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, make_scorer
from .model_config import MODEL_REGISTRY
from ..logs import logger

logger.setup_logger()
logger = logging.getLogger(__name__)

class GridTuner:
    """
    Tối ưu tham số bằng Grid Search.
    """
    def __init__(self, X, y, seed=42):
        self.X = X
        self.y = y
        self.seed = seed
        self.n_classes = len(np.unique(y))

    def _build_scorer(self, scoring, use_predict_proba):
        """
        Tính điểm số để đánh giá.

        Tham số đầu vào:
            - scoring: chỉ số đánh giá.
            - use_predict_proba: có sử dụng predict proba hay không (tính roc_auc).

        Trả về điểm số.
        """
        if scoring == "roc_auc":
            if not use_predict_proba:
                raise ValueError("roc_auc requires predict_proba")

            if self.n_classes == 2:
                return make_scorer(
                    roc_auc_score,
                    needs_proba=True
                )

            return make_scorer(
                roc_auc_score,
                needs_proba=True,
                multi_class="ovr",
                average="macro"
            )

        return make_scorer(
            f1_score,
            average="binary" if self.n_classes == 2 else "macro"
        )

    def optimize(self, model_config, cv=3, scoring=None, n_jobs=-1):
        """
        Hàm tối ưu tham số.

        Tham số đầu vào:
            - model_config: tên model hoặc model đã khởi tạo cần tối ưu.
            - cv: số fold cho cross validation.
            - scoring: chỉ số đánh giá.
            - n_jobs: số luồng làm việc.

        Trả về mô hình với tham số đã tối ưu.
        """
        if isinstance(model_config, str):
            model_config = MODEL_REGISTRY[model_config]

        scoring = scoring or ("roc_auc" if self.n_classes == 2 else "f1")

        logger.info("────────────────────────────────────────────────────────────────────────────────────────────")
        logger.info(f"[GridSearch] Bắt đầu tối ưu tham số mô hình: {model_config.name}")
        logger.info(f"[GridSearch] Scoring: {scoring}")
        logger.info(f"[GridSearch] CV folds: {cv}")

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.seed)

        scorer = self._build_scorer(scoring, model_config.use_predict_proba)

        model = model_config.model_cls(random_state=self.seed)

        grid = GridSearchCV(
            estimator=model,
            param_grid=model_config.param_grid,
            scoring=scorer,
            cv=skf,
            n_jobs=n_jobs,
            refit=True,
            verbose=1
        )

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            grid.fit(self.X, self.y)

        # Logging chi tiết kết quả từng tổ hợp
        logger.info("Kết quả từng combination:")
        for i, params in enumerate(grid.cv_results_['params']):
            mean_score = grid.cv_results_['mean_test_score'][i]
            logger.info(f"[GridSearch] Trial {i+1}: score={mean_score:.4f}, params={params}")

        # Best result
        logger.info(f"[GridSearch] Best Params: {grid.best_params_}")
        logger.info(f"[GridSearch] Best CV Score: {grid.best_score_:.4f}")
        logger.info("────────────────────────────────────────────────────────────────────────────────────────────")

        return grid.best_estimator_