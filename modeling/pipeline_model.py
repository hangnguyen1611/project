import pandas as pd
import joblib
import numpy as np
import logging
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

from ..logs import logger
from .model_trainer import ModelTrainer
from .grid_tuner import GridTuner
from .optuna_tuner import OptunaTuner
from .best_model_selector import BestModelSelector
from .evaluator import EvaluateModel
from .explainer import SHAPExplainer

logger.setup_logger()
logger = logging.getLogger(__name__)

class ModelTrainPipeline:
    """
    Pipeline cho việc train model.
    """
    def __init__(self, random_seed=42, scaler=None, encoders=None, num_cols=None):
        self.df = None
        self.target = None
        self.model = None
        self.shap_explainer = None
        self.scaler = scaler
        self.encoders = encoders
        self.num_cols = num_cols

        # Reproducibility
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    # ------------------ 1. Load dữ liệu ------------------ #
    def load_data(self, data=None, target=None):
        """
        Nạp dữ liệu đã xử lý.

        Tham số đầu vào:
            - data: Tên file dữ liệu đã xử lý hoặc DataFrame đã xử lý.
            - target: Biến mục tiêu.
        """
        # Kiểm tra data được truyền vào là tên file hay là DataFrame
        if isinstance(data, str):
            input_path = Path(data)
            if input_path.exists() or input_path.is_absolute() or '/' in data or '\\' in data:
                file_path = input_path
            else:
                current_file_path = Path(__file__)
                file_path = current_file_path.parent.parent / 'data' / input_path

            self.df = pd.read_csv(file_path)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("data phải là CSV file path hoặc DataFrame.")

        # Kiểm tra tên cột target hợp lệ
        if target in self.df.columns:
            self.target = target
        else:
            raise ValueError(f"Cột mục tiêu '{target}' không tồn tại trong DataFrame.")

        return self
    
    # ------------------ 2. Chia tập dữ liệu ------------------ #
    def split_data(self, test_size=0.2, stratify=True):
        """
        Chia tập dữ liệu thành train/test.

        Tham số đầu vào:
            - test_size: kích cỡ của tập test (mặc định 20%).
            - stratify: giữ tỉ lệ phân bố nhãn như ban đầu trong cả train và test.
        """
        # Transform dữ liệu
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        stratify_param = y if stratify else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=self.random_seed,
            stratify=stratify_param
        )
        
        return self

    # ------------------ 3. Huấn luyện mô hình ------------------ #
    def train(self, model_name=None, model=None):
        """
        Huấn luyện mô hình.

        Tham số đầu vào:
            - model_name: tên model.
            - model: model muốn train.
        """
        trainer = ModelTrainer(model=model, model_name=model_name, seed=self.random_seed)

        self.model = trainer.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        self.shap_explainer = None

        return self

    # ------------------ 4. Tối ưu tham số ------------------ #
    def optimize_params(self, model_name, method="optuna", **kwargs):
        """
        Tối ưu tham số.

        Tham số đầu vào:
            - model_name: tên model muốn tối ưu tham số.
            - method: cách tối ưu (dùng grid search hoặc optuna).
        """
        logger.info(f"Tuning {model_name} with {method}")

        if method == "grid":
            tuner = GridTuner(self.X_train, self.y_train)
            self.model = tuner.optimize(model_name, **kwargs)
        else:
            tuner = OptunaTuner(self.X_train, self.y_train)
            self.model = tuner.optimize(model_name, **kwargs)

        self.shap_explainer = None

        return self
    
    # ------------------ 5. Chọn mô hình tốt nhất ------------------ #
    def select_best_model(self, method="optuna", scoring=None):
        """
        Chọn mô hình tốt nhất.

        Tham số đầu vào:
            - method: cách tối ưu tham số.
            - scoring: chỉ số đánh giá để lựa chọn model.
        """
        selector = BestModelSelector(
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            method=method,
            scoring=scoring,
            random_seed=self.random_seed
        )

        results = selector.select()
        self.model = results["best_model"]
        self.shap_explainer = None

        return results

    # ------------------ 6. Đánh giá mô hình ------------------ #
    def evaluate(self, encoders=None):
        """
        Đánh giá mô hình.
        """
        evaluator = EvaluateModel(self.X_test, self.y_test, self.model)
        evaluator.plot_confusion_matrix(encoders=encoders, target=self.target)
        return evaluator.evaluate()

    # ------------------ 7. Giải thích mô hình ------------------ #
    def explain(self):
        """
        Khởi tạo SHAPExplainer.
        """
        if self.model is None:
            raise ValueError("Chưa có model để explain")

        if self.shap_explainer is None:
            self.shap_explainer = SHAPExplainer(
                model=self.model,
                X_train=self.X_train,
                X_test=self.X_test,
                encoders=self.encoders,
                scaler=self.scaler,
                num_cols=self.num_cols
            )

        return self.shap_explainer
    
    def shap_beeswarm(self):
        """
        Vẽ SHAP beeswarm.
        """
        return self.explain().beeswarm()

    def shap_dependence(self):
        """
        Vẽ SHAP dependence plot.
        """
        return self.explain().dependence()

    def shap_force(self, sample_index=0):
        """
        Vẽ force plot cho sample_index (mẫu) cụ thể.
        """
        return self.explain().force(sample_index)

    # ------------------ 8. Save và Load mô hình------------------ #
    def save(self, output_path_str="model.pkl"):
        """
        Lưu mô hình bằng joblib.

        Tham số đầu vào:
            - path: tên file hoặc đường dẫn lưu file.
        """
        output_path = Path(output_path_str)
        if output_path.exists() or output_path.is_absolute() or '/' in output_path_str or '\\' in output_path_str:
            file_path = output_path
        else:
            current_file_path = Path(__file__)
            file_path = current_file_path.parent.parent / 'modeling' / output_path
        joblib.dump(self.model, file_path)

    def load(self, input_path_str):
        """
        Nạp mô hình bằng joblib.

        Tham số đầu vào:
            - path: tên file hoặc đường dẫn file.
        """
        input_path = Path(input_path_str)
        if input_path.exists() or input_path.is_absolute() or '/' in input_path_str or '\\' in input_path_str:
            input_path_str = input_path
        else:
            current_file_path = Path(__file__)
            file_path = current_file_path.parent.parent / 'modeling' / input_path
        joblib.dump(self.model, file_path)
        self.model = joblib.load(file_path)
        return self.model
