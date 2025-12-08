import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DataVisualizer:
    def __init__(self, data):
        self.data = data
        self.numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def _subplot(self, columns, plot_func, n_cols=3, figsize=(15, 10)):
        n = len(columns)
        rows = (n + n_cols - 1) // n_cols

        plt.figure(figsize=figsize)
        for i, col in enumerate(columns, 1):
            plt.subplot(rows, n_cols, i)
            plot_func(col)
            plt.title(col)
        plt.tight_layout()
        plt.show()

    def histogram(self):
        """
        Tạo các histogram cho các cột dữ liệu số.
        """
        self._subplot(self.numeric_cols,
                      lambda col: sns.histplot(self.data[col], kde=True, edgecolor="black"))

    def barplot(self):
        """
        Vẽ barplot cho các cột phân loại.
        """
        self._subplot(self.categorical_cols,
                      lambda col: sns.countplot(y=self.data[col], hue=self.data[col], edgecolor="black", palette="pastel", legend=False))
        
    def boxplot_for_numeric_cols(self, title=None):
        numeric_df = self.data[self.numeric_cols]   # dataframe chỉ chứa các cột số

        n = len(self.numeric_cols)
        rows = (n + 2) // 3                         # 3 biểu đồ mỗi hàng
        plt.figure(figsize=(14, 4 * rows))

        for i, col in enumerate(self.numeric_cols, 1):
            plt.subplot(rows, 3, i)
            numeric_df.boxplot(column=col)
            plt.title(col)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    # ------------------ Kiểm tra tương quan ------------------ #
    def heatmap(self):
        """
        Vẽ heatmap cho tất cả các cột (dùng với dữ liệu đã encode). 
        """
        plt.figure(figsize=(14, 10))

        sns.heatmap(
            self.data.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5
        )

        plt.title("Heatmap cho bộ dữ liệu", fontsize=18)
        plt.tight_layout()
        plt.show()

    def scatter(self, target=None):
        """
        Vẽ scatter giữa các cột số với target
        """
        cols = [c for c in self.numeric_cols if c != target]
        y = self.data[target].values

        plt.figure(figsize=(15, 5))

        n = len(self.numeric_cols)
        rows = (n + 2) // 3                         # 3 biểu đồ mỗi hàng
        plt.figure(figsize=(14, 4 * rows))

        for i, col in enumerate(cols, 1):
            plt.subplot(rows, 3, i)

            x = self.data[col].values
            plt.scatter(x, y, alpha=0.7)

            # Thêm đường hồi quy
            coeffs = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            plt.plot(xs, np.polyval(coeffs, xs), color="red", linestyle='--')

            plt.title(f"{col} vs {target}")
            plt.xlabel(col)
            plt.ylabel(target)

        plt.tight_layout()
        plt.show()

    def heatmap_one_column(self, target=None):
        """
        Vẽ heatmap dạng 1 cột thể hiện tương quan giữa các đặc trưng số và target.
        """
        # Lấy các cột số để tính tương quan
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Tính tương quan với target
        corr = self.data[numeric_cols].corr()[[target]].sort_values(by=target, ascending=False)

        plt.figure(figsize=(10, 5))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            cbar=True
        )

        plt.title(f"Tương quan với {target}", fontsize=13)
        plt.xlabel("")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()