# ĐỒ ÁN: DỰ ĐOÁN NGUY CƠ MẮC CÁC VẤN ĐỀ LIÊN QUAN ĐẾN SỨC KHỎE TINH THẦN

## 1. Xác định bài toán
* Lĩnh vực: Sức khỏe.
* Loại bài toán chính: Classification.
* Input: Dữ liệu đầu vào bao gồm `gender`, `age`, `employment_status`, `work_environment`,...
* Output: Dữ liệu đầu ra - `target` là `mental_health_risk`.

## 2. Giới thiệu Dataset
| Cột | Mô tả |
|:----|:------|
| **age** | Độ tuổi |
| **gender** | Giới tính (Male - nam,  Female - nữ) |
| **employment_status** | Tình trạng việc làm |
| **work_environment** | Môi trường làm việc |                  
| **mental_health_history** | Có lịch sử mắc vấn đề về sức khỏe tinh thần |          
| **seeks_treatment** | Có tìm đến điều trị hay không |
| **stress_level** | Mức độ căng thẳng (từ 1 đến 10) |
| **sleep_hour** | Số giờ ngủ mỗi ngày |
| **physical_activity_days** | Số ngày vận động mỗi tuần |
| **depression_score** | Điểm đo lường triệu chứng trầm cảm |                  
| **anxiety_score** | Điểm đo lường độ mức lo âu |  
| **social_support_score** | Điểm đánh giá mức độ hỗ trợ xã hội (gia đình, bạn bè,...) |  
| **productivity_score** | Đánh giá năng suất làm việc |  
| **mental_health_risk** | Nguy cơ sức khỏe tinh thần|  

## 3. Cấu trúc 
```
project/
├── artifacts/                     (Lưu các file/đối tượng tạo ra trong quá trình chạy)
|
├── data/
│   └── mental_health_dataset.csv  
|
├── logs/
│   └── logger.py                  
|
├── modeling/
│   ├── best_model_selector.py     (Chọn mô hình tốt nhất)
│   ├── evaluator.py               (Đánh giá mô hình)
│   ├── explainer.py               (Giải thích mô hình bằng SHAP)
│   ├── grid_tuner.py              (Tối ưu siêu tham số Grid Search)
│   ├── model_config.py            (Cấu hình mô hình)
│   ├── model_trainer.py           (Huấn luyện mô hình)
│   ├── optuna_tuner.py            (Tối ưu siêu tham số Optuna)
│   ├── pipeline_model.py          (Pipeline mô hình)
|   └── runner_train.py
|
├── notebook/
│   └── project.ipynb              (notebook)
|
├── preprocessing/
│   ├── preprocessor.py            (Tiền xử lý dữ liệu)
|   ├── runner_preprocess.py
│   └── visualizer.py              (Trực quan hóa dữ liệu)
|
├── main.py                        (python script)
├── README.md                 
└── requirements.txt              
```                  
## 4. Hướng dẫn cài đặt
```
python --version
python -m venv venv
pip install -r requirements.txt
```
## 5. Hướng dẫn chạy
_**Chạy file main.py**_

Chạy các lệnh sau trên **terminal**:
```
# Tiền xử lý dữ liệu
python -m main --mode preprocess --data mental_health_dataset.csv 

# Huấn luyện mô hình, chọn mô hình tốt nhất
python -m main --mode train --data new_mental_health_dataset.csv
```
_**Chạy trên file project.ipynb để xem quá trình tiền xử lý và huấn luyện mô hình**_

### a. Import Module
```python
import sys
from pathlib import Path

project_root = Path.cwd().parent  
sys.path.append(str(project_root))

from preprocessing.preprocessor import DataPreprocessor
from preprocessing.visualizer import DataVisualizer
from modeling.pipeline_model import ModelTrainPipeline
```

### b. Tiền xử lý dữ liệu
```python
d = DataPreprocessor.load('mental_health_dataset.csv')
d.summary()
```
_File dữ liệu sau khi tiền xử lý sẽ được lưu tại thư mục `data`._
### c. Huấn luyện và đánh giá mô hình
```python
pipe = (
    ModelTrainPipeline(random_seed=42, scaler=scaler, encoders=encoders, num_cols=num_cols)
    .load_data(data="new_mental_health_dataset.csv", target="mental_health_risk")
    .split_data()
)

# Huấn luyện mô hình
pipe.train(model_name="logistic")

# Đánh giá 
pipe.evaluate(encoders=encoders)
```
### d. Chọn mô hình tốt nhất và tối ưu tham số
```python
# Chọn mô hình tốt nhất
best_model = pipe.select_best_model()
best_model = pipe.select_best_model(method="grid")

# Tối ưu tham số
pipe.optimize_params()
pipe.optimize_params(method="grid")
```
### e. Giải thích model với SHAP
```python
# Vẽ SHAP plot
pipe.shap_beeswarm()
pipe.shap_dependence()
pipe.shap_force(sample_index=200)
```
_Kết quả train model sẽ được xuất file và lưu tại thư mục `artifacts`._
### f. Theo dõi log
Các thông báo về quá trình _train model_ sẽ được lưu trữ tại thư mục `logs`.
