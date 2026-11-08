# 🫁 Nghiên Cứu Chẩn Đoán Viêm Phổi (Pneumonia) Đa Phương Thức Trên Tập Dữ Liệu SimSUM

Dự án này tập trung nghiên cứu và cải tiến **khâu tiền xử lý dữ liệu (Data Preprocessing)** trên tập dữ liệu y tế đa phương thức **SimSUM** (kết hợp giữa hồ sơ bệnh án dạng Bảng và Ghi chú lâm sàng tự do dạng Văn bản), nhằm giải quyết bài toán phát hiện ca bệnh hiếm **Viêm phổi (`pneu`)**.

---

## 📌 1. Bối Cảnh & Bài Toán Cốt Lõi

* **Tập dữ liệu:** SimSUM (10.000 hồ sơ bệnh nhân mô phỏng bệnh đường hô hấp).
* **Mục tiêu:** Chỉ tập trung dự đoán nhãn **Viêm phổi (`pneu` = 1)**, không dự đoán cảm cúm thông thường (`common_cold`).
* **Thách thức lớn nhất:** 
  * **Mất cân bằng dữ liệu cực đoan:** Chỉ có **103 ca Viêm phổi trên 10.000 hồ sơ (tỷ lệ 1.03%)**, tức tỷ lệ âm tính / dương tính lên tới 96:1.
  * **Xung đột & chồng lấn dữ liệu bảng:** 71% số dòng trong bảng bị trùng lặp đặc trưng (nhiều ca có chỉ số giống hệt nhau nhưng một người bị viêm phổi, người kia thì không). Điều này khẳng định **bắt buộc phải khai thác thêm thông tin văn bản lâm sàng (clinical text)**.
* **Nguyên tắc thực nghiệm công bằng:**
  * Giữ cố định một mô hình phân loại duy nhất: **Logistic Regression (`class_weight='balanced'`)**.
  * Mọi sự cải thiện về điểm số đều chứng minh được **giá trị từ các kỹ thuật tiền xử lý dữ liệu** chứ không phải do chọn mô hình phức tạp hơn.
  * Đánh giá chuẩn y tế: **F1-score (Pneumonia), Recall, Precision, PR-AUC, ROC-AUC** thông qua **5-Fold Stratified Cross-Validation**.

---

## 🗂️ 2. Cấu Trúc Thư Mục & Vai Trò Các File

```text
code/
│
├── 01_split_data.py               # Script lọc cột rò rỉ và chia Train (80%) / Test (20%)
├── train.csv                      # Tập huấn luyện (8.000 dòng, 82 ca viêm phổi)
├── test.csv                       # Tập kiểm thử độc lập (2.000 dòng, 21 ca viêm phổi)
│
├── 02_eda_train.ipynb             # File Jupyter Notebook phân tích khám phá dữ liệu (EDA) trên tập Train
│
├── kich_ban_1_tabular.py          # Kịch bản 1: Baseline chỉ dùng dữ liệu Bảng
├── kich_ban_2_naive_fusion.py     # Kịch bản 2: Baseline đa phương thức - Ghép thô Bảng + Text (785-d)
├── kich_ban_3_balanced_compression.py # Kịch bản 3: Đột phá tiền xử lý Nén PCA (32-d) + Cân bằng mẫu SMOTE
│
├── train_text_emb.npy             # Vector embedding 768-d của tập Train (tạo từ BioLORD-2023)
├── test_text_emb.npy              # Vector embedding 768-d của tập Test
│
└── plots/                         # Thư mục lưu trữ biểu đồ và báo cáo metrics
    ├── eda_target_imbalance.png
    ├── eda_cramers_v_features.png
    ├── eda_n_symptoms_distribution.png
    ├── kich_ban_1_ket_qua.png     # Đồ thị 4 góc đánh giá Kịch bản 1
    ├── kich_ban_2_ket_qua.png     # Đồ thị 4 góc đánh giá Kịch bản 2
    ├── kich_ban_3_ket_qua.png     # Đồ thị 4 góc đánh giá Kịch bản 3
    └── bang_so_sanh_cac_kich_ban.csv # Bảng tổng hợp đối chiếu kết quả các kịch bản
```

---

## ⚙️ 3. Quy Trình Tiền Xử Lý Dữ Liệu Đã Thực Hiện

### A. Lọc bỏ các cột tránh rò rỉ dữ liệu (Anti Data Leakage)
1. **`antibiotics` & `days_at_home`:** Bỏ hoàn toàn vì đây là thông tin sau khám/điều trị (hệ quả điều trị), nếu để vào mô hình sẽ "học vẹt".
2. **`common_cold`:** Bỏ vì không còn là mục tiêu chẩn đoán.
3. **`policy` & `self_empl`:** Bỏ theo khuyến nghị kịch bản thực tế của bài báo gốc SimSUM và kiểm định Chi-square cho thấy hoàn toàn không có ý nghĩa thống kê đối với viêm phổi ($p > 0.30$).

### B. Kỹ thuật đặc trưng mới (Feature Engineering từ EDA)
* Tạo thêm cột **`n_symptoms` (Tổng số triệu chứng xuất hiện đồng thời)**:  
  EDA chỉ ra rằng khi bệnh nhân chỉ có 0–2 triệu chứng thì tỷ lệ viêm phổi $< 1\%$, nhưng khi có đủ 4–5 triệu chứng thì tỷ lệ nhảy vọt lên **$30\%$**. Đây là biến phi tuyến tính quan trọng giúp phân tách ca bệnh nặng.

---

## 🧪 4. Chi Tiết Các Kịch Bản Thực Nghiệm

### 📍 Kịch bản 1: Tabular-only Baseline (`kich_ban_1_tabular.py`)
* **Dữ liệu:** Chỉ dùng 11 cột đặc trưng dạng bảng (`dysp`, `cough`, `pain`, `nasal`, `smoking`, `asthma`, `COPD`, `hay_fever`, `fever`, `season`, `n_symptoms`).
* **Tiền xử lý:** 
  * Biến nhị phân: `OrdinalEncoder` ($0/1$).
  * Biến thứ bậc `fever`: `OrdinalEncoder` (`none` < `low` < `high`).
  * Biến danh mục `season`: `OneHotEncoder`.
  * `n_symptoms`: `StandardScaler`.
* **Mô hình:** `LogisticRegression(class_weight='balanced')`.
* **Hiện tượng:** Mô hình bắt được nhiều ca bệnh (16/21 ca) nhưng số lượng báo động giả quá cao (170 ca bị đoán nhầm thành viêm phổi) do dữ liệu bảng bị trùng lặp tổ hợp.

### 📍 Kịch bản 2: Naive Early Fusion (`kich_ban_2_naive_fusion.py`)
* **Dữ liệu:** Bảng (11 chiều) + Vector văn bản thô từ mô hình y sinh **BioLORD-2023** (768 chiều).
* **Tiền xử lý:** Ghép nối trực tiếp thành ma trận **779 chiều**.
* **Kết quả & Thất bại:** Tái hiện đúng thất bại của các nghiên cứu đi trước: Số chiều văn bản (768) áp đảo hoàn toàn số chiều bảng (11) $\rightarrow$ mô hình bị "rối" và bỏ sót hơn nửa số ca bệnh (chỉ bắt được 10/21 ca viêm phổi).

### 📍 Kịch bản 3: Nén chiều Hình học + Cân bằng mẫu SMOTE (`kich_ban_3_balanced_compression.py`)
* **Dữ liệu:** Bảng (11 chiều) + Vector văn bản đã nén.
* **Đột phá tiền xử lý:**
  1. **Khử lệch nón (Anisotropy):** Trừ vector trung bình của tập Train (*Centering*) + Chuẩn hóa độ dài vector (*$L_2$ Normalization*).
  2. **Nén chiều PCA:** Giảm từ 768 chiều xuống **32 chiều** (giữ lại ~85% ngữ nghĩa quan trọng, tạo sự cân bằng tỷ lệ với nhánh Bảng).
  3. **Cân bằng mẫu SMOTE:** Áp dụng thuật toán SMOTE trên không gian vector kết hợp (chỉ áp dụng trên tập Train/từng fold Train) để nâng tỷ lệ lớp thiểu số lên 20%.
* **Kết quả:** Bắt lại đủ **16/21 ca bệnh**, đồng thời **giảm 43% số ca báo động nhầm (từ 170 ca xuống 97 ca)** so với Kịch bản 1!

---

## 🚀 5. Hướng Dẫn Chạy Dự Án Cho Thành Viên Nhóm

Mọi kịch bản đều được tối ưu hóa để **chạy trực tiếp trên CPU máy tính văn phòng thông thường** (chỉ mất vài giây cho mỗi file).

1. **Cài đặt thư viện cần thiết:**
   ```powershell
   pip install pandas numpy scikit-learn matplotlib seaborn sentence-transformers imbalanced-learn
   ```

2. **Chạy từng kịch bản:**
   * Chạy Kịch bản 1: `python kich_ban_1_tabular.py`
   * Chạy Kịch bản 2: `python kich_ban_2_naive_fusion.py`
   * Chạy Kịch bản 3: `python kich_ban_3_balanced_compression.py`

3. **Xem kết quả:**
   * Mở file **`plots/bang_so_sanh_cac_kich_ban.csv`** để xem bảng điểm tổng hợp đối chiếu.
   * Mở thư mục **`plots/`** để lấy các file ảnh kết quả 4 góc dán thẳng vào slide báo cáo!
