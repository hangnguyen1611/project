from .preprocessor import DataPreprocessor
import joblib

def run_preprocessing(input_csv: str):
    pre = DataPreprocessor.load(input_csv)
    pre.convert("age")
    pre.create_new_feature()
    pre.feature_separation()
    pre.handle_missing_numeric()
    pre.handle_missing_categorical()
    pre.handle_outlier()
    pre.normalize_categorical()

    encoders = pre.encode_categorical()
    scaler = pre.scale_features()

    pre.summary()

    # Lưu dữ liệu processed
    pre.new_data()
    print("Dữ liệu sau xử lý đã được lưu.")

    # Lưu encoders + scaler
    joblib.dump(encoders, "artifacts/encoders.pkl")
    joblib.dump(scaler, "artifacts/scaler.pkl")
    print("Encoders & scaler lưu tại artifacts/")
