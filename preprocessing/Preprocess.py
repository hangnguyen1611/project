import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest

class DataPreprocessor:
    def __init__(self, data=None, missing_strategy='median', outlier_type='IQR', scaler_type='standard', encoding_type='label'):
        self.data = data 

        self.missing_strategy = missing_strategy
        self.outlier_type = outlier_type
        self.scaler_type = scaler_type
        self.encoding_type = encoding_type

        self.categorical_cols = []
        self.numeric_cols = []

        self.scaler = StandardScaler()
        self.minmax = MinMaxScaler()

        self.encoders = {}
    
    def __repr__(self):
        if self.data is not None:
            return f"{self.data.shape[0]} rows, {self.data.shape[1]} cols"
        else:
            return "No data loaded"
    
    # METHOD
    @staticmethod
    def is_numeric(col):
        return pd.api.types.is_numeric_dtype(col)

    # ------------------1.1 Load dữ liệu------------------ #
    @classmethod 
    def load(cls, input_path_str = 'mental_health_dataset.csv'):
        input_path = Path(input_path_str)
        if input_path.exists() or input_path.is_absolute() or '/' in input_path_str or '\\' in input_path_str:
            file_path = input_path
        else:
            current_file_path = Path(__file__)
            file_path = current_file_path.parent.parent / 'data' / input_path
            
        try:
            df = pd.read_csv(file_path)
            return cls(df) 
        except Exception as e:
            print("Lỗi khi đọc file:", e)
            return None  
        
    # ------------------1.2 Convert data type------------------ #
    def convert(self):
        print(self.data.dtypes)
        self.data['age'] = pd.to_numeric(self.data['age'], errors='coerce')

    # ------------------1.3 Phân loại các cột------------------ #
    def feature_separation(self):
        # Chia các cột thành 2 phần: cột kiểu dữ liệu phân loại và cột kiểu dữ liệu số
        self.categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        self.numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()

        print("Các cột phân loại:", self.categorical_cols)
        print("Các cột số:", self.numeric_cols)

        return self.categorical_cols, self.numeric_cols

    # ------------------1.4 Thông tin về dữ liệu------------------ #
    def summary(self):
        print("\nKích thước dữ liệu:", self.data.shape)
        print("\nMô tả dữ liệu:")
        print(self.data.info())
        print(self.data.describe().T)

    # ------------------2. Xử lý giá trị khuyết------------------ #
    def handle_missing_numeric(self):
        for col in self.numeric_cols:
            if self.missing_strategy == 'median':
                self.data[col] = self.data[col].fillna(self.data[col].median())
            elif self.missing_strategy == 'mean':
                 self.data[col] = self.data[col].fillna(self.data[col].mean())
            elif self.missing_strategy == 'drop':
                self.data[col] = self.data.dropna(subset=[col])
        return self

    def handle_missing_categorical(self):
        for col in self.categorical_cols:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0]) # thay các giá trị NaN bằng giá trị xuất hiện nhiều nhất
        return self
    
    # ------------------3. Chuẩn hóa dữ liệu cho các cột categorical------------------ #
    def normalize_categorical(self):
        for col in self.categorical_cols:
            print("********************")
            print(self.data[col].value_counts())
        print("\n*****************Sau khi xử lí*****************\n")
        
        normalize_col = {'gender': {'male': 'Male',
                                    'M': 'Male',
                                    'female': 'Female',
                                    'FM': 'Female',
                                    'F': 'Female',
                                    'Non-binary': 'Other',
                                    'Prefer not to say': 'Other',
                                    'Non ': 'Other'},
                        'employment_status': {'employed': 'Employed'},
                        'work_environment': {'on-site': 'On-site',
                                            'hybrid': 'Hybrid'},
                        'mental_health_history': {'N': 'No',
                                                'Y': 'Yes'},
                        'seeks_treatment': {'N': 'No',
                                            'no': 'No',
                                            'yes': 'Yes',
                                            'Y': 'Yes'}}

        for col, normal_col in normalize_col.items():
            self.data[col] = self.data[col].replace(normal_col)
            print(self.data[col].value_counts())
            print("********************")

        return self
    
    # ------------------4. Xử lý ngoại lai------------------ #
    def skewness_for_numeric_cols(self):
        numeric_df = self.data[self.numeric_cols]   # dataframe chỉ chứa các cột số
        skewness = numeric_df.skew()
        print("Skew cho các cột:")
        print(skewness)
        
    def handle_outlier(self):
        numeric_df = self.data[self.numeric_cols]   # dataframe chỉ chứa các cột số

        for col in numeric_df:
            if self.outlier_type == 'IQR':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1 
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                self.data.loc[self.data[col] < lower, col] = lower
                self.data.loc[self.data[col] > upper, col] = upper

            elif self.outlier_type == 'Z-score':
                mean = self.data[col].mean()
                std = self.data[col].std()

                z_score = (self.data[col] - mean) / std
                threshold = 3

                self.data.loc[z_score < -threshold, col] = mean - threshold*std
                self.data.loc[z_score > threshold, col] = mean + threshold*std
            
            else:
                iso = IsolationForest(contamination=0.03)
                predictions = iso.fit_predict(self.data[[col]])

                #outlier_index = np.where(predictions==-1)
                self.data.loc[predictions==-1, col] = self.data[col].median()

        return self 

    # ---------------------- 5. Encode và Scale --------------------------------
    def encode_categorical(self):
        self.encoders = {}

        if self.encoding_type == 'label':
            for col in self.categorical_cols:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])

                self.encoders[col] = {
                    "encoder": le,
                    "columns": [col]
                }

        elif self.encoding_type == 'onehot':
            ohe = OneHotEncoder(sparse_output=False)
            encoded = ohe.fit_transform(self.data[self.categorical_cols])

            encoded_cols = ohe.get_feature_names_out(self.categorical_cols)
            df_encoded = pd.DataFrame(encoded, columns=encoded_cols)

            self.data = pd.concat([
                self.data.drop(columns=self.categorical_cols),
                df_encoded
            ], axis=1)

            self.encoders["onehot"] = {
                "encoder": ohe,
                "orig_cols": self.categorical_cols,
                "encoded_cols": encoded_cols.tolist()
            }

        return self.encoders
    
    def scale_features(self):
        if self.scaler_type == 'standard':
            self.data[self.numeric_cols] = self.scaler.fit_transform(self.data[self.numeric_cols])
            return self.scaler 
        else:
            self.data[self.numeric_cols] = self.minmax.fit_transform(self.data[self.numeric_cols])
            return self.minmax

    # ------------------ 6. Tạo đặc trưng mới------------------ #
    def create_new_feature(self):
        # Đánh giá hiệu suất làm việc dựa trên mức stress
        self.data['productivity_base_on_stress'] = self.data['productivity_score'] / self.data['stress_level']
        self.numeric_cols.append("productivity_base_on_stress")
        return self
    
    # ------------------ 7. Trả về dữ liệu sau khi tiền xử lý ------------------ #
    def get_processed_data(self):
        return self.data
    
    def new_data(self, file_name="new_mental_health_dataset.csv"):
        self.data.to_csv(file_name, index=False)