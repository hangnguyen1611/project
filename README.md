<<<<<<< HEAD
# ĐỒ ÁN: DỰ ĐOÁN NGUY CƠ MẮC CÁC VẤN ĐỀ LIÊN QUAN ĐẾN SỨC KHỎE TINH THẦN

## 1. Giới thiệu đồ án
Thực trạng về sức khỏe tinh thần đang là một vấn đề đáng báo động trên toàn cầu. Theo Tổ chức Y tế Thế giới (WHO), ước tính có gần một tỷ người trên thế giới đang sống chung với chứng rối loạn tâm thần. Cụ thể hơn, tại Việt Nam, theo thống kê, khoảng 15% dân số mắc các rối loạn tâm thần phổ biến, trong đó có tới ba triệu người bị rối loạn trầm cảm. 

Dựa vào bối cảnh đó, nhóm lựa chọn ứng dụng các mô hình để dự đoán nguy cơ mắc các vấn đề liên quan sức khỏe tinh thần dựa vào các yếu tố như tuổi tác, giới tính, môi trường làm việc,... Việc dự đoán này không chỉ hỗ trợ trong việc phát hiện và can thiệp sớm mà còn giảm thiểu gánh nặng y tế.

### Xác định bài toán
* Lĩnh vực: Sức khỏe
* Loại bài toán chính: Classification
* Input: Dữ liệu đầu vào bao gồm `gender`, `age`, `employment_status`, `work_environment`,...
* Output: Dữ liệu đầu ra - target là `mental_health_risk`.

## 2. Giới thiệu Dataset
| Cột | Mô tả |
|:----|:------|
| **age** | Độ tuổi |
| **gender** | Giới tính ( Male - nam,  Female - nữ) |
| **employment_status** | Tình trạng việc làm |
| **work_environment** | Môi trường làm việc |                  
| **mental_health_history** | Có lịch sử mắc vấn đề về sức khỏe tinh thần |          
| **seeks_treatment** | Có tìm đến điều trị hay không |
| **stress_level** | Mức độ căng thẳng (từ 1 đến 10) |
| **sleep_hour** | Số giờ ngủ mỗi ngày |
| **physical_activity_days** | Số ngày vận động mỗi tuần |
| **depression_score** | Điểm đo lường triệu chứng trầm cảm |                  
| **anxiety_score** | Điểm đo lường độ mức lo âu |  
| **social_support_score** | Điểm đánh giá mức độ hỗ trợ xã hội(gia đình, bạn bè,...) |  
| **productivity_score** | Đánh giá năng suất làm việc |  
| **mental_health_risk** | Nguy cơ sức khỏe tinh thần|  

## 3. Cấu trúc 
```
Project/
│
├── README.md
│
├── requirements.txt
│
├── data/
│   └─ mental_health_dataset.csv
│
├── src/ 
│   ├── Preprocess.py                   
│   └── ModelTrainer.py   
│              
├── output/
│   ├── new_mental_health_dataset.csv   
│   ├── Project.ipynb 
```                  
## 4. Hướng dẫn cài đặt
```python
python --version
python -m venv venv
pip install -r requirements.txt
```
## 5. Hướng dẫn chạy
_Chạy trên file Project.ipynb_

### a. Import Module
```python
from Model import ModelTrainer
from Preprocess import DataPreprocessor
```

### b. Tiền xử lý dữ liệu
```python
d = DataPreprocessor.load('mental_health_dataset.csv')
d.summary()
```
### c. Huấn luyện và đánh giá mô hình
```python
trainer_grid = ModelTrainer(random_seed=42, preprocessor=d)
trainer_grid.load_data("new_mental_health_dataset.csv", target='mental_health_risk').head()

optimized_model = trainer_grid.optimize_params(cv=3, scoring='accuracy')

best_model = trainer_grid.train_model(metric='accuracy')
```
### d. Tối ưu siêu tham số
```python
optimized_model = trainer_grid.optimize_params(cv=3, scoring='accuracy')
```
### e. Giải thích giá trị
```python
explain["shap_values"]
```
=======
# Python
>>>>>>> e875a084c5131ab6d43caeb6588725d235b334f0
