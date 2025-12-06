import sys
sys.path.append('../../')
from modeling import *

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# -------------------- Logging ----------------------
# Tắt log Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Thiết lập Root Logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Xóa tất cả StreamHandler
root_logger.handlers = [h for h in root_logger.handlers if not isinstance(h, logging.StreamHandler)]

# Thêm FileHandler nếu chưa tồn tại
if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
    fh = logging.FileHandler('model_training.log', mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(fh)

# -------------------- Model ----------------------
class ModelTrainer:
    """
    Lớp này dùng để huấn luyện mô hình
    """
    def __init__(self, data, target, random_seed=42):
        self.df = data
        self.target = target
        self.model = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Reproducibility
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    # ------------------1. Nạp dữ liệu------------------ #
    @classmethod
    def load_data(cls, data, target):
        """
        Nạp dữ liệu đã xử lý.

        Tham số đầu vào:
            - filename: Tên file dữ liệu đã xử lý hoặc DataFrame đã xử lý.
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

            df = pd.read_csv(file_path)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("data phải là CSV file path hoặc DataFrame.")

        # Kiểm tra tên cột target hợp lệ
        if target not in df.columns:
            raise ValueError(f"Cột mục tiêu '{target}' không tồn tại trong DataFrame.")

        return cls(df, target)
    
    # ------------------2. Chia tập dữ liệu------------------ #
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
    
    # ------------------3. Huấn luyện mô hình------------------ #
    def train_model(self, model_type="logistic"):
        """
        Huấn luyện mô hình theo loại mô hình lựa chọn.

        Tham số đầu vào:
            - model_type: loại model (logistic|randomforest|lightgbm|xgboost).
        """  
        logging.info(f"Training {model_type}...")

        model_map = {
            "logistic": LogisticRegression(
                max_iter=2000, 
                class_weight="balanced", 
                random_state=self.random_seed
            ),
            "randomforest": RandomForestClassifier(
                n_estimators=200, 
                random_state=self.random_seed
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=500, 
                learning_rate=0.05, 
                random_state=self.random_seed, 
                verbosity=-1
            ),
            "xgboost": XGBClassifier(
                n_estimators=300, 
                learning_rate=0.1, 
                random_state=self.random_seed, 
                verbosity=0
            )
        }

        if model_type not in model_map:
            raise ValueError(f"model_type phải thuộc: {list(model_map.keys())}")

        self.model = model_map[model_type]

        if model_type == "lightgbm":
            self.model.fit(
                self.X_train, self.y_train, 
                eval_set=[(self.X_test, self.y_test)]
            )
        elif model_type == "xgboost":
            self.model.fit(
                self.X_train, self.y_train, 
                eval_set=[(self.X_test, self.y_test)], 
                verbose=False
            )
        else:
            self.model.fit(self.X_train, self.y_train)

        logging.info(f"{model_type} trained successfully.")
        return self
        
    # ------------------4. Tối ưu hóa tham số------------------ #
    def objective(self, trial: optuna.Trial, model_type="logistic", cv=3, scoring=None, n_classes=None) -> float:
        """
        Objective function cho Optuna.

        Tham số đầu vào:
            - trial: Đối tượng của Optuna.
            - model_type: loại model muốn tối ưu (logistic/randomforest/lightgbm/xgboost).
            - cv: số fold cho Cross Validation.
            - scoring: chỉ số đánh giá.
            - n_classes: số lớp của target.
        
        Trả về: Điểm số trung bình.
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_seed)
        scores = []

        # ------------ Thiết lập object và tính scale_pos_weight ------------ 
        if n_classes == 2:
            lgbm_objective = "binary"
            neg = (self.y_train == 0).sum()
            pos = (self.y_train == 1).sum()
            scale_pos_weight = neg / pos if pos > 0 else 1.0
        
        elif n_classes > 2:
            lgbm_objective = "multiclass"
            scale_pos_weight = 1.0        # Không dùng trong Multiclass

        # ------------ LightGBM ------------ 
        if model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "max_depth": trial.suggest_categorical("max_depth",[-1, 3, 5, 7, 9]),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
                "bagging_freq": 1,
                "objective": lgbm_objective,
                "num_class": n_classes if n_classes > 2 else None,
                "scale_pos_weight": scale_pos_weight,
                "random_state": self.random_seed,
                "n_jobs": 1,
                "verbosity": -1
            }
            Estimator = LGBMClassifier

            pruning_callback = None
            if scoring == "roc_auc":
                pruning_callback = [LightGBMPruningCallback(trial, "auc" if n_classes == 2 else "auc_mu")]

        # ------------ XGBoost ------------ 
        elif model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "scale_pos_weight": scale_pos_weight,
                "tree_method": "hist",
                "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
                "random_state": self.random_seed,
                "verbosity": 0,
                "n_jobs": 1
            }
            if n_classes > 2:
                params.update({
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "num_class": n_classes
                })
            else:
                params.update({
                    "objective": "binary:logistic",
                    "eval_metric": "logloss"
                })
            Estimator = XGBClassifier
            pruning_callback = None

        # ------------ Random Forest ------------ 
        elif model_type == "randomforest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 25),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": self.random_seed,
                "n_jobs": 1
            }
            Estimator = RandomForestClassifier
            pruning_callback = None
            
        # ------------ Logistic Regression ------------ 
        else: 
            params = {
                "C": trial.suggest_float("C", 1e-3, 100, log=True),
                "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
                "max_iter": 2000,
                "random_state": self.random_seed
            }
            Estimator = LogisticRegression
            pruning_callback = None

        # ------------ Cross validation ------------ 
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):

            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            model = Estimator(**params)

            if model_type == "lightgbm":
                model.fit(
                    X_tr, y_tr, 
                    eval_set=[(X_val, y_val)],
                    callbacks=pruning_callback 
                )
            elif model_type == "xgboost":
                model.fit(
                    X_tr, y_tr, 
                    eval_set=[(X_val, y_val)], 
                    verbose=False
                )
            else:
                model.fit(X_tr, y_tr)

            # ------------ Đánh giá ------------ 
            if scoring == "roc_auc":
                if n_classes == 2:
                    proba = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, proba)
                else:
                    proba = model.predict_proba(X_val)
                    score = roc_auc_score(y_val, proba, multi_class="ovr")
            elif scoring == "f1":
                pred = model.predict(X_val)
                score = f1_score(y_val, pred, average="binary" if n_classes==2 else "macro")
            elif scoring == "recall":
                pred = model.predict(X_val)
                score = recall_score(y_val, pred, average="binary" if n_classes==2 else "weighted")
            else:
                pred = model.predict(X_val)
                score = accuracy_score(y_val, pred)

            scores.append(score)

            # ------------ Pruning thủ công ------------ 
            if pruning_callback is None:
                trial.report(score, step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_score = float(np.mean(scores))
        logging.info(f"Trial {trial.number} - params: {params} - mean_score: {mean_score}")
        return mean_score
        
    def optimize_params_with_optuna(self, model_type="logistic", n_trials=30, cv=3, scoring=None, timeout=None, use_pruner=True):  
        """
        Tối ưu hóa tham số với Optuna.

        Tham số đầu vào:
            - model_type: loại model (logistic | randomforest | lightgbm | xgboost).
            - n_trials: số lần thử của Optuna.
            - cv: số fold trong StratifiedKFold.
            - scoring: rou_auc | f1 | recall | accuracy | None (Mặc định roc_auc cho binary, f1 cho multiclass).
            - timeout: giới hạn thời gian học.
            - user_pruner: nếu True bật pruning để dừng trial kém sớm.

        Trả về dict chưa:
            - best_model: estimator đã fit trên tập train.
            - study: đối tượng optuna.study.Study.
            - best_params: tham số tốt nhất.
        """ 
        logging.info("Tối ưu tham số với Optuna...")

        n_classes = len(np.unique(self.df[self.target]))

        # Chọn scoring
        if scoring is None:
            scoring = "roc_auc" if n_classes == 2 else "f1"

        sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30) if use_pruner else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        # Chạy study
        optimize_fn = lambda trial: self.objective(trial, model_type=model_type, cv=cv, scoring=scoring, n_classes=n_classes)

        if timeout:
            study.optimize(optimize_fn, n_trials=n_trials, timeout=timeout)
        else:
            study.optimize(optimize_fn, n_trials=n_trials)

        best_params = study.best_trial.params
        logging.info(f"Best params ({model_type}): {best_params}")

        # ------------ Fit lại model tốt nhất trên toàn bộ X_train ------------ 
        if model_type == "lightgbm":
            lgb_params = best_params.copy()
            lgb_params.update({"random_state": self.random_seed, "n_jobs": 1, "verbose": -1})
            model = LGBMClassifier(**lgb_params)
            model.fit(self.X_train, self.y_train)

        elif model_type == "xgboost":
            xgb_params = best_params.copy()
            xgb_params.update({"random_state": self.random_seed, "verbosity": 0, "n_jobs": 1})
            model = XGBClassifier(**xgb_params)
            model.fit(self.X_train, self.y_train)

        elif model_type == "randomforest":
            rf_params = best_params.copy()
            rf_params.update({"random_state": self.random_seed, "n_jobs":1})
            model = RandomForestClassifier(**rf_params)
            model.fit(self.X_train, self.y_train)
            
        else:  # logistic
            log_params = best_params.copy()
            log_params.update({"random_state": self.random_seed, "max_iter": 2000})
            model = LogisticRegression(**log_params)
            model.fit(self.X_train, self.y_train)

        return {
            "best_model": model,
            "study": study,
            "best_params": best_params
        }
    
    def optimize_params_with_grid_search(self, param_grid=None, cv=3, scoring='accuracy'):
        """
        Tối ưu siêu tham số bằng GridSearchCV.

        Tham số đầu vào:
            - param_grid: lưới tham số.
            - cv: số fold cho Cross Validation.
            - scoring: chỉ số đánh giá.
        
        Trả về: model đã tối ưu.
        """
        if self.model is None:
            raise ValueError("Gọi train_model trước khi tối ưu GridSearch!")
        
        logging.info("Tối ưu tham số với GridSearch...")

        if param_grid is None:
            if isinstance(self.model, LogisticRegression):
                param_grid = {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']  # liblinear hỗ trợ l1/l2
                }
            elif isinstance(self.model, RandomForestClassifier):
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif isinstance(self.model, XGBClassifier):
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 1.0]
                }
            elif isinstance(self.model, LGBMClassifier):
                param_grid = {
                    'num_leaves': [31, 63, 127],
                    'max_depth': [ -1, 10],
                    'learning_rate': [0.01, 0.05],
                    'subsample': [0.7, 0.9],
                    'colsample_bytree': [0.7, 1.0],
                    'bagging_freq': [1]
                }
            else:
                raise ValueError(f"Không hỗ trợ GridSearch cho model loại {type(self.model)}")
        
        grid = GridSearchCV(self.model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
        grid.fit(self.X_train, self.y_train)

        self.model = grid.best_estimator_
        logging.info(f"Best params: {grid.best_params_}, best CV score: {grid.best_score_:.4f}")

        return self.model
    
    # ------------------5. Chọn mô hình tốt nhất------------------ #
    def best_model(self, method="grid", scoring=None):
        """
        Chọn ra mô hình tốt nhất và tối ưu tham số.

        Tham số đầu vào:
            - method: cách tối ưu tham số (grid/optuna).
            - scoring: chỉ số đánh giá model.
        
        Trả về dict chứa:
            - best_model_type: loại mô hình tốt nhất.
            - best_model: mô hình tốt nhất đã fit.
            - all_metrics: đánh giá cho tất cả các mô hình.
        """
        logging.info("Chọn mô hình tốt nhất...")

        models = {
            'logistic': LogisticRegression(
                max_iter=1000,
                random_state=self.random_seed
            ),
            'randomforest': RandomForestClassifier(
                random_state=self.random_seed,
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                random_state=self.random_seed
            ),
            "lightgbm": LGBMClassifier(
                random_state=self.random_seed, 
                verbosity=-1
            )
        }

        best_score = -float('inf')
        best_model_type = None
        all_metrics = []

        # Chọn scoring
        n_classes = len(np.unique(self.y_train))
        if scoring is None:
            scoring = "roc_auc" if n_classes == 2 else "f1"

        # Duyệt từng model
        for model_type, model in models.items():
            logging.info(f"Training {model_type}...")
            model.fit(self.X_train, self.y_train)

            # Đánh giá mô hình hiện tại
            self.model = model
            metrics_df = self.evaluate()
            metrics_df["model_type"] = model_type
            all_metrics.append(metrics_df)
            
            # Tính score
            if scoring == "roc_auc":
                if "roc_auc" in metrics_df.columns:
                    score = metrics_df["roc_auc"].iloc[0]
                elif "roc_auc_ovr" in metrics_df.columns:
                    score = metrics_df["roc_auc_ovr"].iloc[0]
                else:
                    score = metrics_df["f1"].iloc[0]    # fallback
            else:
                score = metrics_df["f1"].iloc[0]

            logging.info(f"{model_type} score: {score:.4f}")

            # Chọn model tốt nhất
            if score > best_score:
                best_score = score
                best_model = model
                best_model_type = model_type

        logging.info(f"Mô hình được chọn: {best_model_type}, score = {best_score:.4f}")

        # Tối ưu hóa siêu tham số cho best model
        self.model = best_model
        if method == "grid":
            self.model = self.optimize_params_with_grid_search()
        else:
            optimized = self.optimize_params_with_optuna(model_type=best_model_type)
            self.model = optimized["best_model"]

        # Đánh giá cho tất cả mô hình
        all_metrics = pd.concat(all_metrics, ignore_index=True)
        cols = ["model_type"] + [c for c in all_metrics.columns if c != "model_type"]
        all_metrics = all_metrics[cols]

        logging.info(f"Best model: {best_model_type} with optimized params.")

        return {
            "best_model_type": best_model_type,
            "best_model": self.model,
            "all_metrics": all_metrics
        }
    
    # ------------------6. Đánh giá mô hình------------------ #
    def evaluate(self):
        """
        Đánh giá mô hình.

        Trả về df metrics bao gồm: accuracy, precision, recall, f1 và roc-auc(roc_auc_ovr).
        """
        y_pred = self.model.predict(self.X_test)

        # Xác định binary hay multiclass
        is_binary = self.y_test.nunique() == 2
        avg_method = "binary" if is_binary else "macro"

        # Đánh giá
        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average=avg_method, zero_division=0),
            "recall": recall_score(self.y_test, y_pred, average=avg_method, zero_division=0),
            "f1": f1_score(self.y_test, y_pred, average=avg_method, zero_division=0)
        }

        # ROC-AUC
        metrics["roc_auc"] = None
        metrics["roc_auc_ovr"] = None

        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(self.X_test)
                if is_binary and proba.shape[1] == 2: # binary
                    metrics["roc_auc"] = roc_auc_score(self.y_test, proba[:, 1])
                else:  # multiclass
                    metrics["roc_auc_ovr"] = roc_auc_score(self.y_test, proba, multi_class="ovr")
        except Exception as e:
            logging.warning(f"Không thể tính ROC-AUC: {e}")
        
        return pd.DataFrame({k:[v] for k, v in metrics.items()})
    
    def visualize_confusion_matrix(self):
        """
        Ma trận nhầm lẫn (confusion).
        """
        # Giá trị dự đoán
        y_pred = self.model.predict(self.X_test)

        # Confusion_matrix
        cm = confusion_matrix(self.y_test, y_pred) 
        labels = np.unique(self.y_test)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    # ------------------ 7. Giải thích mô hình với SHAP (Feature Importance) ------------------ #
    def explain_with_shap(self):
        """
        Giải thích mô hình bằng SHAP.

        Trả về dict gồm: explainer và shap_values.
        """
        shap.initjs()

        if isinstance(self.model, LogisticRegression):
            explainer = shap.LinearExplainer(self.model, self.X_train)
            shap_values = explainer.shap_values(self.X_test)
        else:
            explainer = shap.TreeExplainer(self.model, feature_perturbation="tree_path_dependent")
            try:
                shap_values = explainer.shap_values(self.X_test)
            except:
                shap_values = explainer.shap_values(self.X_test.values)

        return {
            "explainer": explainer,
            "shap_values": shap_values
        }
    
    # ------------------ 7.1. Vẽ SHAP beeswarm ------------------ #
    def shap_beeswarm(self, sample_index=0):
        """
        Vẽ SHAP beeswarm plot.

        Tham số đầu vào: 
            - sample_index: Chỉ số mẫu.
        """
        result = self.explain_with_shap()
        explainer = result['explainer']
        shap_values = result['shap_values']

        if isinstance(shap_values, list):
            class_id = np.argmax(self.model.predict_proba(self.X_test.iloc[[sample_index]]))
            shap_vals = shap_values[class_id]
            base_value = explainer.expected_value[class_id]
        elif shap_values.ndim == 3:
            class_id = np.argmax(self.model.predict_proba(self.X_test.iloc[[sample_index]]))
            shap_vals = shap_values[:, :, class_id]
            base_value = explainer.expected_value[class_id]
        else:
            shap_vals = shap_values
            base_value = explainer.expected_value

        print("====================== BEESWARM ======================")
        shap.plots.beeswarm(
            shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=self.X_test,
                feature_names=self.X_test.columns,
            ),
            max_display=20
        )

    # ------------------7.2. Vẽ SHAP dependence plot------------------ #
    def shap_dependece_plot(self, sample_index=0):
        """
        Vẽ SHAP dependence plot.

        Tham số đầu vào: 
            - sample_index: Chỉ số mẫu.
        """
        result = self.explain_with_shap()
        shap_values = result['shap_values']

        # Lấy SHAP values cho lớp mục tiêu
        if isinstance(shap_values, list):
            class_id = np.argmax(self.model.predict_proba(self.X_test.iloc[[sample_index]]))
            shap_vals = np.array(shap_values[class_id])
        elif shap_values.ndim == 3:
            class_id = np.argmax(self.model.predict_proba(self.X_test.iloc[[sample_index]]))
            shap_vals = shap_values[:, :, class_id]
        else:
            shap_vals = shap_values

        print("======================== DEPENDENCE PLOTS ========================")
        features = self.X_test.columns
        n = len(features)
        ncol = 3
        nrow = (n + ncol - 1) // ncol

        # Tạo figure lớn
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(18, 6*nrow))
        axes = axes.flatten()
        original_figure = plt.figure    # lưu bản gốc

        def no_new_fig(*args, **kwargs):
            return fig                  # luôn trả về figure đang dùng
        plt.figure = no_new_fig

        for i, feature in enumerate(features):
            ax = axes[i]
            plt.sca(axes[i])
            shap.dependence_plot(
                feature,
                shap_vals,
                self.X_test,
                interaction_index=feature,
                show=False
            )
            ax.set_title(f"Dependence Plot: {feature}")

        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.figure = original_figure
        plt.show()
    
    # ------------------ 7.3. Vẽ SHAP force plot ------------------ #
    def inverse_row(self, row_enc, encoders, scaler, num_cols):
        row_disp = row_enc.copy()

        # Inverse LabelEncoder
        for col, info in encoders.items():
            if col == "onehot":
                continue

            enc = info["encoder"]

            if col in row_disp.index:
                try:
                    row_disp[col] = enc.inverse_transform([int(row_disp[col])])[0]
                except:
                    pass

        # Inverse OneHotEncoder
        if "onehot" in encoders:
            ohe = encoders["onehot"]["encoder"]
            enc_cols = encoders["onehot"]["encoded_cols"]
            orig_cols = encoders["onehot"]["orig_cols"]

            onehot_vector = row_enc[enc_cols].values.reshape(1, -1)
            orig_vals = ohe.inverse_transform(onehot_vector)[0]

            for i, c in enumerate(orig_cols):
                row_disp[c] = orig_vals[i]

        # Inverse Scaler
        if scaler is not None:
            scaled_vals = row_enc[num_cols].values.reshape(1, -1)

            inverse_vals = scaler.inverse_transform(scaled_vals)[0]

            for i, c in enumerate(num_cols):
                row_disp[c] = inverse_vals[i]

        return row_disp

    def shap_force_plot(self, sample_index=0, encoders=None, scaler=None, num_cols=None, cat_cols=None):
        """
        Vẽ SHAP force plot.

        Tham số đầu vào: 
            - sample_index: Chỉ số mẫu muốn vẽ force plot (mặc định mẫu đầu tiên).
            - scaler: Scaler để inverse X_test về data gốc.
            - num_cols: Danh sách các cột đã được scale.
        """
        result = self.explain_with_shap()
        explainer = result['explainer']
        shap_values = result['shap_values']

        # Lấy SHAP values cho lớp mục tiêu
        if isinstance(shap_values, list):
            class_id = np.argmax(self.model.predict_proba(self.X_test.iloc[[sample_index]]))
            sample_shap = shap_values[class_id][sample_index]
            base_value = explainer.expected_value[class_id]
        elif shap_values.ndim == 3:
            class_id = np.argmax(self.model.predict_proba(self.X_test.iloc[[sample_index]]))
            sample_shap = shap_values[sample_index, :, class_id]
            base_value = explainer.expected_value[class_id]
        else:
            sample_shap = shap_values[sample_index]
            base_value = explainer.expected_value
        
        # Lấy lại data gốc                                
        row_disp = self.inverse_row(self.X_test.iloc[sample_index], encoders, scaler, num_cols)    # giá trị gốc

        print(f"=============== SHAP Force Plot cho sample index = {sample_index} ===============")
        shap.force_plot(
            base_value,
            sample_shap,
            features=row_disp,
            matplotlib=True,
            show=True
        )
    
    # ------------------8. Save và Load model------------------ #
    def save_model(self, filename):
        """
        Lưu mô hình.

        Tham số đầu vào:
            - filename: Tên file lưu.
        """
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        """
        Nạp mô hình.

        Tham số đầu vào:
            - filename: Tên file model cần load.
        
        Trả về model đã load.
        """
        self.model = joblib.load(filename)
        return self.model