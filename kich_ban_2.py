import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Đảm bảo thư mục plots đã có
os.makedirs("plots", exist_ok=True)

# ==========================================
# 1. ĐỌC DỮ LIỆU BẢNG & TRÍCH XUẤT EMBEDDING
# ==========================================
print("1. Đang đọc train.csv và test.csv...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Feature Engineering: n_symptoms
bin_symptoms = ["dysp", "cough", "pain", "nasal"]


def add_features(df):
    df_new = df.copy()
    df_new["n_symptoms"] = (df_new[bin_symptoms] == "yes").sum(axis=1) + (
        df_new["fever"] != "none"
    ).astype(int)
    return df_new


train_df = add_features(train_df)
test_df = add_features(test_df)

y_train = train_df["pneu"]
y_test = test_df["pneu"]

# ==========================================
# 2. XỬ LÝ NHÁNH VĂN BẢN (BIOLORD EMBEDDING)
# ==========================================
# Kiểm tra xem đã có file embedding lưu từ trước chưa để đỡ phải tính lại
train_emb_file = "train_text_emb.npy"
test_emb_file = "test_text_emb.npy"

if os.path.exists(train_emb_file) and os.path.exists(test_emb_file):
    print("-> Tìm thấy file embedding đã lưu, đang tải trực tiếp vào RAM...")
    X_train_text = np.load(train_emb_file)
    X_test_text = np.load(test_emb_file)
else:
    print(
        "-> Chưa có file embedding, đang tải mô hình BioLORD-2023 để mã hóa (chỉ chạy 1 lần)..."
    )
    embedder = SentenceTransformer("FremyCompany/BioLORD-2023")

    print("Mã hóa văn bản tập Train (768 chiều)...")
    X_train_text = embedder.encode(
        train_df["advanced_text"].tolist(), show_progress_bar=True
    )

    print("Mã hóa văn bản tập Test (768 chiều)...")
    X_test_text = embedder.encode(
        test_df["advanced_text"].tolist(), show_progress_bar=True
    )

    # Lưu lại để các kịch bản sau dùng tiếp mà không cần chạy lại bước này
    np.save(train_emb_file, X_train_text)
    np.save(test_emb_file, X_test_text)
    print("=> Đã lưu embedding vào file .npy thành công!")

print(
    f"Kích thước ma trận văn bản: Train={X_train_text.shape}, Test={X_test_text.shape}"
)

# ==========================================
# 3. TIỀN XỬ LÝ NHÁNH BẢNG (TABULAR PREPROCESSING)
# ==========================================
binary_cols = [
    "dysp",
    "cough",
    "pain",
    "nasal",
    "smoking",
    "asthma",
    "COPD",
    "hay_fever",
]
ordinal_cols = ["fever"]
nominal_cols = ["season"]
numeric_cols = ["n_symptoms"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "bin",
            OrdinalEncoder(categories=[["no", "yes"]] * len(binary_cols)),
            binary_cols,
        ),
        (
            "ord",
            OrdinalEncoder(categories=[["none", "low", "high"]]),
            ordinal_cols,
        ),
        ("nom", OneHotEncoder(drop="first"), nominal_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Fit preprocessor CHỈ TRÊN TẬP TRAIN để tránh Data Leakage
X_train_tab = preprocessor.fit_transform(train_df)
X_test_tab = preprocessor.transform(test_df)
print(
    f"Kích thước ma trận bảng: Train={X_train_tab.shape}, Test={X_test_tab.shape}"
)

# ==========================================
# 4. GHÉP NỐI THÔ SƠ (NAIVE EARLY FUSION)
# ==========================================
# Ghép ma trận Bảng (11-d) và ma trận Text (768-d) -> 779 chiều
X_train_fused = np.hstack([X_train_tab, X_train_text])
X_test_fused = np.hstack([X_test_tab, X_test_text])
print(
    f"-> Kích thước ma trận sau ghép nối (Fused): {X_train_fused.shape} (779 chiều)"
)

# ==========================================
# 5. ĐÁNH GIÁ 5-FOLD STRATIFIED CV TRÊN TRAIN
# ==========================================
print("\n" + "=" * 55)
print("--- BẮT ĐẦU 5-FOLD STRATIFIED CV (KỊCH BẢN 2) ---")
print("=" * 55)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1, cv_rec, cv_prec, cv_roc, cv_prauc = [], [], [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_fused, y_train), 1):
    X_tr, y_tr = X_train_fused[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train_fused[val_idx], y_train.iloc[val_idx]

    model = LogisticRegression(
        class_weight="balanced", random_state=42, max_iter=1000
    )
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    cv_f1.append(f1_score(y_val, y_val_pred, pos_label=1))
    cv_rec.append(recall_score(y_val, y_val_pred, pos_label=1))
    cv_prec.append(precision_score(y_val, y_val_pred, pos_label=1, zero_division=0))
    cv_roc.append(roc_auc_score(y_val, y_val_proba))
    cv_prauc.append(average_precision_score(y_val, y_val_proba))

print(f"KẾT QUẢ TRUNG BÌNH 5-FOLD (VALIDATION - KB 2):")
print(f"-> F1-score (Pneumonia) : {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")
print(f"-> Recall (Độ nhạy)     : {np.mean(cv_rec):.4f} ± {np.std(cv_rec):.4f}")
print(f"-> Precision (Độ chính xác): {np.mean(cv_prec):.4f} ± {np.std(cv_prec):.4f}")
print(f"-> ROC-AUC              : {np.mean(cv_roc):.4f} ± {np.std(cv_roc):.4f}")
print(f"-> PR-AUC               : {np.mean(cv_prauc):.4f} ± {np.std(cv_prauc):.4f}")

# ==========================================
# 6. ĐÁNH GIÁ TRÊN TẬP TEST (2.000 MẪU)
# ==========================================
print("\n" + "=" * 55)
print("--- ĐÁNH GIÁ TRÊN TẬP TEST (KỊCH BẢN 2) ---")
print("=" * 55)

final_model = LogisticRegression(
    class_weight="balanced", random_state=42, max_iter=1000
)
final_model.fit(X_train_fused, y_train)

y_test_pred = final_model.predict(X_test_fused)
y_test_proba = final_model.predict_proba(X_test_fused)[:, 1]

test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
test_rec = recall_score(y_test, y_test_pred, pos_label=1)
test_prec = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")
test_roc = roc_auc_score(y_test, y_test_proba)
test_prauc = average_precision_score(y_test, y_test_proba)

print(f"1. F1-score (Pneumonia) : {test_f1:.4f}")
print(f"2. Recall (Độ nhạy)     : {test_rec:.4f}")
print(f"3. Precision (Độ chuẩn) : {test_prec:.4f}")
print(f"4. Macro-F1             : {test_macro_f1:.4f}")
print(f"5. ROC-AUC              : {test_roc:.4f}")
print(f"6. PR-AUC               : {test_prauc:.4f}")

print("\nBáo cáo phân loại chi tiết:")
print(
    classification_report(
        y_test, y_test_pred, target_names=["Không bệnh (0)", "Viêm phổi (1)"]
    )
)

# ==========================================
# 7. VẼ TOÀN BỘ ĐỒ THỊ VÀO 1 FILE ẢNH
# ==========================================
print("\nĐang xuất biểu đồ kết quả...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "KỊCH BẢN 2: ĐÁNH GIÁ GHÉP NỐI THÔ SƠ BẢNG + VĂN BẢN (NAIVE EARLY FUSION)",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

# Hình 1: Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    ax=axes[0, 0],
    xticklabels=["Không bệnh (0)", "Viêm phổi (1)"],
    yticklabels=["Không bệnh (0)", "Viêm phổi (1)"],
    annot_kws={"size": 13, "weight": "bold"},
)
axes[0, 0].set_title(
    "1. Ma trận nhầm lẫn (Confusion Matrix)", fontsize=11, fontweight="bold"
)
axes[0, 0].set_ylabel("Nhãn thực tế")
axes[0, 0].set_xlabel("Nhãn dự đoán")

# Hình 2: Đường cong ROC
RocCurveDisplay.from_predictions(
    y_test, y_test_proba, ax=axes[0, 1], name="Naive Fusion"
)
axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
axes[0, 1].set_title(
    f"2. Đường cong ROC (AUC = {test_roc:.4f})", fontsize=11, fontweight="bold"
)
axes[0, 1].grid(True, linestyle="--", alpha=0.6)
axes[0, 1].legend(loc="lower right")

# Hình 3: Đường cong PR
PrecisionRecallDisplay.from_predictions(
    y_test, y_test_proba, ax=axes[1, 0], name="Naive Fusion"
)
axes[1, 0].set_title(
    f"3. Đường cong PR (PR-AUC = {test_prauc:.4f})",
    fontsize=11,
    fontweight="bold",
)
axes[1, 0].grid(True, linestyle="--", alpha=0.6)
axes[1, 0].legend(loc="upper right")

# Hình 4: So sánh độ lớn trọng số giữa Nhánh Bảng vs Nhánh Văn bản
coefs = final_model.coef_[0]
tab_coef_mag = np.mean(np.abs(coefs[: X_train_tab.shape[1]]))
text_coef_mag = np.mean(np.abs(coefs[X_train_tab.shape[1] :]))

mag_df = pd.Series(
    [tab_coef_mag, text_coef_mag],
    index=["Đặc trưng Bảng (11-d)", "Đặc trưng Text BioLORD (768-d)"],
)
mag_df.plot(kind="bar", ax=axes[1, 1], color=["#2A9D8F", "#E76F51"], edgecolor="black")
axes[1, 1].set_title(
    "4. Độ lớn trọng số trung bình: Bảng vs Văn bản",
    fontsize=11,
    fontweight="bold",
)
axes[1, 1].set_ylabel("Giá trị tuyệt đối hệ số trung bình (|coef|)")
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)
axes[1, 1].grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = os.path.join("plots", "kich_ban_2_ket_qua.png")
plt.savefig(plot_path, dpi=300)
plt.close()

# ==========================================
# 8. LƯU THỐNG KÊ VÀO FILE TỔNG HỢP METRICS
# ==========================================
new_row = pd.DataFrame(
    [
        {
            "Kịch bản": "KB 2 (Naive Fusion)",
            "F1-score (Pneumonia)": round(test_f1, 4),
            "Recall": round(test_rec, 4),
            "Precision": round(test_prec, 4),
            "Macro-F1": round(test_macro_f1, 4),
            "ROC-AUC": round(test_roc, 4),
            "PR-AUC": round(test_prauc, 4),
            "CV_F1_mean": round(np.mean(cv_f1), 4),
            "CV_F1_std": round(np.std(cv_f1), 4),
        }
    ]
)

metrics_file = os.path.join("plots", "bang_so_sanh_cac_kich_ban.csv")
if os.path.exists(metrics_file):
    all_metrics = pd.read_csv(metrics_file)
    all_metrics = pd.concat([all_metrics, new_row], ignore_index=True).drop_duplicates(
        subset=["Kịch bản"], keep="last"
    )
else:
    # Nếu chưa có thì lấy từ KB1 qua ghép vào
    kb1_file = os.path.join("plots", "kich_ban_1_metrics.csv")
    if os.path.exists(kb1_file):
        kb1_df = pd.read_csv(kb1_file)
        all_metrics = pd.concat([kb1_df, new_row], ignore_index=True)
    else:
        all_metrics = new_row

all_metrics.to_csv(metrics_file, index=False)

print(f"\n=> ĐÃ XUẤT THÀNH CÔNG HÌNH VẼ: {plot_path}")
print(f"=> ĐÃ CẬP NHẬT BẢNG SO SÁNH METRICS: {metrics_file}")
