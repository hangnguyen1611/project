import pandas as pd
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

from ..logs import logger
logger.setup_logger()
logger = logging.getLogger(__name__)

class EvaluateModel:
    """
    Lớp này dùng để đánh giá mô hình.
    """
    def __init__(self, X_test, y_test, model, average_multiclass="macro"):
        self.X = X_test
        self.y = y_test
        self.model = model
        self.avg_multi = average_multiclass

    def evaluate(self):
        """
        Đánh giá cho mô hình.

        Trả về: DataFrame bao gồm các chỉ số accuracy | precision | recall | f1 | roc_auc | roc_auc_ovr
        """
        logger.info("────────────────────────────────────────────────────────────────────────────────────────────")
        logger.info(f"[Evaluate] Bắt đầu đánh giá mô hình {self.model.__class__.__name__}...")
        y_pred = self.model.predict(self.X)
        is_binary = self.y.nunique() == 2
        avg = "binary" if is_binary else self.avg_multi

        metrics = {
            "accuracy": accuracy_score(self.y, y_pred),
            "precision": precision_score(self.y, y_pred, average=avg, zero_division=0),
            "recall": recall_score(self.y, y_pred, average=avg, zero_division=0),
            "f1": f1_score(self.y, y_pred, average=avg, zero_division=0)
        }

        # ROC-AUC
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(self.X)
                if is_binary and proba.shape[1] == 2: # binary
                    metrics["roc_auc"] = roc_auc_score(self.y, proba[:, 1])
                else:  # multiclass
                    metrics["roc_auc_ovr"] = roc_auc_score(self.y, proba, multi_class="ovr")
        except Exception as e:
            raise RuntimeError(f"Không thể tính ROC-AUC: {e}")
        
        logger.info(f"[Evaluate] Kết quả đánh giá: {metrics}")
        logger.info("────────────────────────────────────────────────────────────────────────────────────────────")

        return pd.DataFrame([metrics])

    def plot_confusion_matrix(self, encoders=None, target=None):
        """
        Vẽ ma trận nhầm lẫn.
        """
        if not encoders or target not in encoders:
            y_true = self.y
            y_pred = self.model.predict(self.X)
        else:
            info = encoders[target]
            enc = info["encoder"]

            try:
                y_true = enc.inverse_transform(self.y.astype(int))
                y_pred = enc.inverse_transform(self.model.predict(self.X).astype(int))
            except Exception:
                y_true = self.y
                y_pred = self.model.predict(self.X)

        labels = sorted(set(y_true) | set(y_pred))

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def comparison_bar_plot(self, model_metrics):
        """
        Vẽ bar plot so sánh metrics giữa các mô hình.

        Tham số đầu vào:
            - model_metrics: Dict[str, pd.DataFrame] chứa các chỉ số đánh giá của từng mô hình.
        """
        metrics_used = ["roc_auc_ovr", "f1", "recall"]

        scores = {m: [] for m in metrics_used}

        for metrics in model_metrics.values():
            for m in metrics_used:
                scores[m].append(metrics.loc[0, m])

        x = np.arange(len(model_metrics))
        width = 0.25
        colors = ["#AEC7E8", "#FFBB78", "#98DF8A"]

        plt.figure(figsize=(8,5))
        for i, metric in enumerate(metrics_used):
            plt.bar(x + i*width, scores[metric], width, label=metric.upper(), color=colors[i], edgecolor="black")
            for j, v in enumerate(scores[metric]):
                plt.text(j + i*width, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

        plt.xticks(x + width, model_metrics.keys())
        plt.ylabel("Score")
        plt.title("So sánh các model bằng biểu đồ cột")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def radar_plot(self, model_metrics, y_lim=(0.96, 1)):
        """
        Vẽ radar plot so sánh các mô hình.

        Tham số đầu vào:
            - model_metrics: Dict[str, pd.DataFrame] chứa các chỉ số đánh giá của từng mô hình.
            - y_lim: Tuple xác định giới hạn trục y.
        """
        labels = ["accuracy", "precision", "recall", "f1", "roc_auc_ovr"]
        num_metrics = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]   # khép kín radar

        plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, polar=True)

        cmap = plt.get_cmap("Set2")   
        colors = cmap.colors

        for i, (model_name, df_metrics) in enumerate(model_metrics.items()):
            values = df_metrics.loc[0, labels].values.tolist()
            values += values[:1]

            ax.plot(angles, values, linewidth=1, label=model_name, marker="o", color=colors[i % len(colors)])

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(y_lim)
        ax.set_title("So sánh các model bằng biểu đồ radar", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

        plt.tight_layout()
        plt.show()