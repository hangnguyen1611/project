import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import warnings

class SHAPExplainer:
    """
    Giải thích mô hình với SHAP bằng các đồ thị SHAP bao gồm:
        - beeswarm plot
        - dependence plot
        - force plot cho mẫu cụ thể
    """
    def __init__(self, model, X_train, X_test, encoders=None, scaler=None, num_cols=None):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.encoders = encoders
        self.scaler = scaler
        self.num_cols = num_cols or []

        shap.initjs()
        self.explainer = self._build_explainer()
        self.shap_values = self._compute_shap()

    def _build_explainer(self):
        """
        Xây dựng explainer.
        """
        if isinstance(self.model, LogisticRegression):
            return shap.LinearExplainer(self.model, self.X_train)
        return shap.TreeExplainer(self.model)

    def _compute_shap(self):
        """
        Tính SHAP value.
        """
        try:
            return self.explainer.shap_values(self.X_test)
        except:
            return self.explainer.shap_values(self.X_test.values)

    def _select_class(self, sample_index=0):
        """
        Chọn lớp cho mẫu.

        Tham số đầu vào:
            - sample_index: chỉ số mẫu
        
        Trả về lớp của mẫu nếu có.
        """
        if hasattr(self.model, "predict_proba"):
            return np.argmax(
                self.model.predict_proba(self.X_test.iloc[[sample_index]])
            )
        return None

    def _get_shap(self, sample_index=0):
        """
        Lấy SHAP value và base value cho mẫu.

        Tham số đầu vào:
            - sample_index: chỉ số mẫu cần lấy SHAP value.
        
        Trả về SHAP value và base value.

        Ghi chú:
            - Với multiclass, phương thức sẽ chọn class trả về SHAP value.
            - Base value là giá trị trung bình dự đoán trước khi biết dữ liệu.
        """
        sv = self.shap_values

        if isinstance(sv, list):               # multiclass (list)
            cid = self._select_class(sample_index)
            return sv[cid], self.explainer.expected_value[cid]

        if sv.ndim == 3:                       # multiclass (3D)
            cid = self._select_class(sample_index)
            return sv[:, :, cid], self.explainer.expected_value[cid]

        return sv, self.explainer.expected_value

    def inverse_row(self, row):
        """
        Lấy lại dữ liệu gốc cho 1 hàng.

        Tham số đầu vào:
            - row: hàng dữ liệu đã mã hóa và chuẩn hóa.
        
        Trả về dữ liệu ban đầu cho hàng đó.
        """
        row_disp = row.copy()

        # ---- LabelEncoder ----
        if self.encoders:
            for col, info in self.encoders.items():
                if col == "onehot":
                    continue
                enc = info["encoder"]
                if col in row_disp.index:
                    try:
                        val = enc.inverse_transform([int(row_disp[col])])[0]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FutureWarning)
                            row_disp[col] = val
                    except:
                        pass

            # ---- OneHotEncoder ----
            if "onehot" in self.encoders:
                ohe = self.encoders["onehot"]["encoder"]
                enc_cols = self.encoders["onehot"]["encoded_cols"]
                orig_cols = self.encoders["onehot"]["orig_cols"]

                orig_vals = ohe.inverse_transform(row[enc_cols].values.reshape(1, -1))[0]

                for c, v in zip(orig_cols, orig_vals):
                    row_disp[c] = v  # gán trực tiếp, không dùng astype

        # ---- Scaler ----
        if self.scaler and self.num_cols:
            scaled_vals = row[self.num_cols].values.reshape(1, -1)
            inverse_vals = self.scaler.inverse_transform(scaled_vals)[0]

            for i, c in enumerate(self.num_cols):
                row_disp[c] = float(inverse_vals[i])

        return row_disp

    def beeswarm(self, max_display=20, sample_index=0):
        """
        Vẽ SHAP beeswarm.

        Tham số đầu vào:
            - sample_index: chỉ số mẫu để vẽ cho class nếu là multiclass.
            - max_display: số feature tối đa muốn hiển thị.
        """
        shap_vals, base = self._get_shap(sample_index=sample_index)

        shap.plots.beeswarm(
            shap.Explanation(
                values=shap_vals,
                base_values=base,
                data=self.X_test,
                feature_names=self.X_test.columns
            ),
            max_display=max_display
        )

    def dependence(self, sample_index=0, ncol=3):
        """
        Vẽ SHAP dependence plot.

        Tham số đầu vào:
            - sample_index: chỉ số mẫu để vẽ cho class nếu là multiclass.
            - ncol: số cột trong figure.
        """
        shap_vals, _ = self._get_shap(sample_index=sample_index)
        features = self.X_test.columns

        n = len(features)
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

    def force(self, sample_index=0):
        """
        Vẽ SHAP force plot cho một mẫu xác định.

        Tham số đầu vào:
            - sample_index: Chỉ số mẫu.
        """
        shap_vals, base = self._get_shap(sample_index)
        sample_shap = shap_vals[sample_index]

        row_disp = self.inverse_row(
            self.X_test.iloc[sample_index]
        )

        shap.force_plot(
            base,
            sample_shap,
            features=row_disp,
            matplotlib=True,
            show=True
        )
