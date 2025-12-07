import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
        metrics["roc_auc"] = None
        metrics["roc_auc_ovr"] = None

        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(self.X)
                if is_binary and proba.shape[1] == 2: # binary
                    metrics["roc_auc"] = roc_auc_score(self.y, proba[:, 1])
                else:  # multiclass
                    metrics["roc_auc_ovr"] = roc_auc_score(self.y, proba, multi_class="ovr")
        except Exception as e:
            raise RuntimeError(f"Không thể tính ROC-AUC: {e}")

        return pd.DataFrame([metrics])

    def plot_confusion_matrix(self):
        """
        Vẽ ma trận nhầm lẫn.
        """
        y_pred = self.model.predict(self.X)
        cm = confusion_matrix(self.y, y_pred)
        labels = np.unique(self.y)

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