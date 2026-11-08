import os
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
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
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    normalize,
)

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

# Tải embedding văn bản 768-d đã tính sẵn từ KB 2
print("-> Đang tải file vector embedding văn bản (768 chiều)...")
X_train_text_raw = np.load("train_text_emb.npy")
X_test_text_raw = np.load("test_text_emb.npy")

# ==========================================
# 2. TIỀN XỬ LÝ HÌNH HỌC VÀ NÉN CHIỀU EMBEDDING (PCA 32-D)
# ==========================================
print("2. Đang thực hiện Centering, L2 Normalization và PCA (768 -> 32 chiều)...")

# Bước 2.1: Trừ tâm (Centering) - Chỉ lấy mean từ tập Train
mean_vec = np.mean(X_train_text_raw, axis=0)
X_train_centered = X_train_text_raw - mean_vec
X_test_centered = X_test_text_raw - mean_vec

# Bước 2.2: Chuẩn hóa L2 Norm
X_train_norm = normalize(X_train_centered, norm="l2")
X_test_norm = normalize(X_test_centered, norm="l2")

# Bước 2.3: Nén chiều bằng PCA (Chỉ fit trên Train)
pca = PCA(n_components=32, random_state=42)
X_train_text_pca = pca.fit_transform(X_train_norm)
X_test_text_pca = pca.transform(X_test_norm)

variance_retained = np.sum(pca.explained_variance_ratio_) * 100
print(f"-> Nén văn bản còn 32 chiều (Giữ lại {variance_retained:.2f}% thông tin ngữ nghĩa)")

# ==========================================
# 3. TIỀN XỬ LÝ DỮ LIỆU BẢNG (TABULAR PREPROCESSING)
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

X_train_tab = preprocessor.fit_transform(train_df)
X_test_tab = preprocessor.transform(test_df)

# ==========================================
# 4. GHÉP VECTOR CÂN ĐỐI (COMPRESSED FUSION)
# ==========================================
# Ghép ma trận Bảng (11-d) + Text PCA (32-d) -> Tổng cộng 43 chiều (Cực kỳ cân đối!)
X_train_fused = np.hstack([X_train_tab, X_train_text_pca])
X_test_fused = np.hstack([X_test_tab, X_test_text_pca])
print(
    f"-> Kích thước ma trận kết hợp sau khi nén: {X_train_fused.shape} (Chỉ còn 43 chiều thay vì 779)"
)

# ==========================================
# 5. ĐÁNH GIÁ 5-FOLD STRATIFIED CV VỚI SMOTE TRÊN TRAIN
# ==========================================
print("\n" + "=" * 55)
print("--- BẮT ĐẦU 5-FOLD STRATIFIED CV KẾT HỢP SMOTE (KỊCH BẢN 3) ---")
print("=" * 55)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1, cv_rec, cv_prec, cv_roc, cv_prauc = [], [], [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_fused, y_train), 1):
    X_tr, y_tr = X_train_fused[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train_fused[val_idx], y_train.iloc[val_idx]

    # Áp dụng SMOTE chỉ trên Fold Train để nâng tỷ lệ lớp 1 lên 20%
    smote = SMOTE(sampling_strategy=0.2, random_state=42)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

    model = LogisticRegression(
        class_weight="balanced", random_state=42, max_iter=1000
    )
    model.fit(X_tr_res, y_tr_res)

    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    cv_f1.append(f1_score(y_val, y_val_pred, pos_label=1))
    cv_rec.append(recall_score(y_val, y_val_pred, pos_label=1))
    cv_prec.append(precision_score(y_val, y_val_pred, pos_label=1, zero_division=0))
    cv_roc.append(roc_auc_score(y_val, y_val_proba))
    cv_prauc.append(average_precision_score(y_val, y_val_proba))

print(f"KẾT QUẢ TRUNG BÌNH 5-FOLD (VALIDATION - KB 3):")
print(f"-> F1-score (Pneumonia) : {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")
print(f"-> Recall (Độ nhạy)     : {np.mean(cv_rec):.4f} ± {np.std(cv_rec):.4f}")
print(f"-> Precision (Độ chính xác): {np.mean(cv_prec):.4f} ± {np.std(cv_prec):.4f}")
print(f"-> ROC-AUC              : {np.mean(cv_roc):.4f} ± {np.std(cv_roc):.4f}")
print(f"-> PR-AUC               : {np.mean(cv_prauc):.4f} ± {np.std(cv_prauc):.4f}")

# ==========================================
# 6. ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP
# ==========================================
print("\n" + "=" * 55)
print("--- ĐÁNH GIÁ TRÊN TẬP TEST (KỊCH BẢN 3) ---")
print("=" * 55)

# Huấn luyện trên toàn bộ Train sau khi chạy SMOTE
smote_final = SMOTE(sampling_strategy=0.2, random_state=42)
X_train_res, y_train_res = smote_final.fit_resample(X_train_fused, y_train)

final_model = LogisticRegression(
    class_weight="balanced", random_state=42, max_iter=1000
)
final_model.fit(X_train_res, y_train_res)

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
# 7. XUẤT HÌNH VẼ ĐÁNH GIÁ RA THƯ MỤC PLOTS
# ==========================================
print("\nĐang xuất biểu đồ kết quả...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "KỊCH BẢN 3: ĐÁNH GIÁ NÉN CHIỀU VĂN BẢN (PCA-32) + CÂN BẰNG MẪU (SMOTE)",
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
    y_test, y_test_proba, ax=axes[0, 1], name="PCA-32 + SMOTE"
)
axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
axes[0, 1].set_title(
    f"2. Đường cong ROC (AUC = {test_roc:.4f})", fontsize=11, fontweight="bold"
)
axes[0, 1].grid(True, linestyle="--", alpha=0.6)
axes[0, 1].legend(loc="lower right")

# Hình 3: Đường cong PR
PrecisionRecallDisplay.from_predictions(
    y_test, y_test_proba, ax=axes[1, 0], name="PCA-32 + SMOTE"
)
axes[1, 0].set_title(
    f"3. Đường cong PR (PR-AUC = {test_prauc:.4f})",
    fontsize=11,
    fontweight="bold",
)
axes[1, 0].grid(True, linestyle="--", alpha=0.6)
axes[1, 0].legend(loc="upper right")

# Hình 4: So sánh độ lớn trọng số giữa Bảng vs Văn bản (đã nén)
coefs = final_model.coef_[0]
tab_coef_mag = np.mean(np.abs(coefs[: X_train_tab.shape[1]]))
text_coef_mag = np.mean(np.abs(coefs[X_train_tab.shape[1] :]))

mag_df = pd.Series(
    [tab_coef_mag, text_coef_mag],
    index=["Đặc trưng Bảng (11-d)", "Đặc trưng Text sau PCA (32-d)"],
)
mag_df.plot(kind="bar", ax=axes[1, 1], color=["#2A9D8F", "#E76F51"], edgecolor="black")
axes[1, 1].set_title(
    "4. Cân bằng trọng số: Bảng vs Văn bản sau PCA",
    fontsize=11,
    fontweight="bold",
)
axes[1, 1].set_ylabel("Hệ số trọng số trung bình (|coef|)")
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)
axes[1, 1].grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = os.path.join("plots", "kich_ban_3_ket_qua.png")
plt.savefig(plot_path, dpi=300)
plt.close()

# ==========================================
# 8. CẬP NHẬT FILE BẢNG TỔNG HỢP METRICS
# ==========================================
new_row = pd.DataFrame(
    [
        {
            "Kịch bản": "KB 3 (PCA-32 + SMOTE)",
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
    all_metrics = new_row

all_metrics.to_csv(metrics_file, index=False)

print(f"\n=> ĐÃ XUẤT HÌNH VẼ: {plot_path}")
print(f"=> ĐÃ CẬP NHẬT BẢNG SO SÁNH METRICS: {metrics_file}")