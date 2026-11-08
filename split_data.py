import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# 1. Đọc dữ liệu từ file SimSUM.csv
path_csv = r"C:\Users\Admin\pptk\SimSUM.csv"
print(f"1. Đang đọc dữ liệu từ: {path_csv}...")
df_raw = pd.read_csv(path_csv, sep=';')
print(f"Kích thước ban đầu: {df_raw.shape}")

# 2. Danh sách các cột cần loại bỏ:
cols_to_drop = [
    'antibiotics',    # Data leakage (thuốc kê sau khám)
    'days_at_home',   # Data leakage (kết quả điều trị)
    'common_cold',    # Bỏ vì bạn chỉ chẩn đoán viêm phổi
    'policy',         # Biến hành chính, không mang ý nghĩa bệnh lý
    'self_empl'       # Biến hành chính, không mang ý nghĩa bệnh lý
]

# Nếu có cột 'Unnamed: 0' (cột index cũ) thì bỏ luôn
if 'Unnamed: 0' in df_raw.columns:
    cols_to_drop.append('Unnamed: 0')

df_cleaned = df_raw.drop(columns=cols_to_drop)

# 3. Chuyển đổi nhãn pneu sang dạng số (1: yes, 0: no)
df_cleaned['pneu'] = (df_cleaned['pneu'] == 'yes').astype(int)

# 4. Chia Train (80%) và Test (20%) có phân tầng (Stratified)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in splitter.split(df_cleaned, df_cleaned['pneu']):
    train_df = df_cleaned.iloc[train_idx].copy()
    test_df = df_cleaned.iloc[test_idx].copy()

# 5. In thông tin kiểm tra
print("\n--- KẾT QUẢ PHÂN CHIA DỮ LIỆU ---")
print(f"-> Tập Train: {len(train_df)} mẫu | Ca Viêm phổi (pneu=1): {train_df['pneu'].sum()} ({train_df['pneu'].mean()*100:.2f}%)")
print(f"-> Tập Test : {len(test_df)} mẫu  | Ca Viêm phổi (pneu=1): {test_df['pneu'].sum()} ({test_df['pneu'].mean()*100:.2f}%)")

# 6. Lưu file vào thư mục code
train_df.to_csv(r"C:\Users\Admin\pptk\code\train.csv", index=False)
test_df.to_csv(r"C:\Users\Admin\pptk\code\test.csv", index=False)

print("\n=> THÀNH CÔNG! Đã tạo 2 file: 'train.csv' và 'test.csv' tại thư mục code.")

print(train_df.head())
