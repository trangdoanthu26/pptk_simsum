import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Tạo thư mục plots nếu chưa có
os.makedirs("plots", exist_ok=True)

# ==========================================
# 1. ĐỌC DỮ LIỆU
# ==========================================
print("Đang đọc dữ liệu train.csv và test.csv...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# ==========================================
# 2. FEATURE ENGINEERING: n_symptoms
# ==========================================
bin_symptoms = ["dysp", "cough", "pain", "nasal"]


def add_features(df):
    df_new = df.copy()
    df_new["n_symptoms"] = (df_new[bin_symptoms] == "yes").sum(axis=1) + (
        df_new["fever"] != "none"
    ).astype(int)
    return df_new


train_df = add_features(train_df)
test_df = add_features(test_df)

ignore_cols = ["pneu", "text", "advanced_text", "Unnamed: 0"]
feature_cols = [c for c in train_df.columns if c not in ignore_cols]

X_train = train_df[feature_cols]
y_train = train_df["pneu"]
X_test = test_df[feature_cols]
y_test = test_df["pneu"]

# ==========================================
# 3. THIẾT LẬP PIPELINE TIỀN XỬ LÝ
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
binary_transformer = OrdinalEncoder(categories=[["no", "yes"]] * len(binary_cols))

ordinal_cols = ["fever"]
ordinal_transformer = OrdinalEncoder(categories=[["none", "low", "high"]])

nominal_cols = ["season"]
nominal_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

numeric_cols = ["n_symptoms"]
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("bin", binary_transformer, binary_cols),
        ("ord", ordinal_transformer, ordinal_cols),
        ("nom", nominal_transformer, nominal_cols),
        ("num", numeric_transformer, numeric_cols),
    ]
)

model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
clf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

# ==========================================
# 4. ĐÁNH GIÁ 5-FOLD STRATIFIED CV TRÊN TRAIN
# ==========================================
print("Đang huấn luyện và đánh giá 5-Fold Stratified CV...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_f1, cv_rec, cv_prec, cv_roc, cv_prauc = [], [], [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

    clf_pipeline.fit(X_tr, y_tr)
    y_val_pred = clf_pipeline.predict(X_val)
    y_val_proba = clf_pipeline.predict_proba(X_val)[:, 1]

    cv_f1.append(f1_score(y_val, y_val_pred, pos_label=1))
    cv_rec.append(recall_score(y_val, y_val_pred, pos_label=1))
    cv_prec.append(precision_score(y_val, y_val_pred, pos_label=1, zero_division=0))
    cv_roc.append(roc_auc_score(y_val, y_val_proba))
    cv_prauc.append(average_precision_score(y_val, y_val_proba))

# ==========================================
# 5. ĐÁNH GIÁ TRÊN TẬP TEST (2.000 MẪU)
# ==========================================
print("Đang đánh giá mô hình trên tập Test...")
clf_pipeline.fit(X_train, y_train)

y_test_pred = clf_pipeline.predict(X_test)
y_test_proba = clf_pipeline.predict_proba(X_test)[:, 1]

test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
test_rec = recall_score(y_test, y_test_pred, pos_label=1)
test_prec = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")
test_roc = roc_auc_score(y_test, y_test_proba)
test_prauc = average_precision_score(y_test, y_test_proba)

# In kết quả dạng số ra terminal
print("\n" + "=" * 50)
print(f"KẾT QUẢ TẬP TEST - KỊCH BẢN 1 (CHỈ DÙNG BẢNG):")
print(f"-> F1-score (Pneumonia) : {test_f1:.4f}")
print(f"-> Recall (Độ nhạy)     : {test_rec:.4f}")
print(f"-> Precision (Độ chuẩn) : {test_prec:.4f}")
print(f"-> Macro-F1             : {test_macro_f1:.4f}")
print(f"-> ROC-AUC              : {test_roc:.4f}")
print(f"-> PR-AUC               : {test_prauc:.4f}")
print("=" * 50)

# ==========================================
# 6. VẼ TOÀN BỘ BIỂU ĐỒ VÀO 1 FILE ẢNH DUY NHẤT
# ==========================================
print("Đang vẽ và xuất toàn bộ đồ thị ra file ảnh...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "KỊCH BẢN 1: ĐÁNH GIÁ MÔ HÌNH CHỈ DÙNG DỮ LIỆU BẢNG (TABULAR-ONLY)",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

# --- Hình 1: Ma trận nhầm lẫn (Confusion Matrix) ---
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
axes[0, 0].set_ylabel("Nhãn thực tế (Ground Truth)")
axes[0, 0].set_xlabel("Nhãn dự đoán (Predicted)")

# --- Hình 2: Đường cong ROC-AUC ---
RocCurveDisplay.from_predictions(
    y_test, y_test_proba, ax=axes[0, 1], name="Logistic Regression"
)
axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance (AUC=0.5)")
axes[0, 1].set_title(
    f"2. Đường cong ROC (AUC = {test_roc:.4f})", fontsize=11, fontweight="bold"
)
axes[0, 1].grid(True, linestyle="--", alpha=0.6)
axes[0, 1].legend(loc="lower right")

# --- Hình 3: Đường cong Precision-Recall (PR-AUC) ---
PrecisionRecallDisplay.from_predictions(
    y_test, y_test_proba, ax=axes[1, 0], name="Logistic Regression"
)
axes[1, 0].set_title(
    f"3. Đường cong PR (PR-AUC = {test_prauc:.4f})",
    fontsize=11,
    fontweight="bold",
)
axes[1, 0].grid(True, linestyle="--", alpha=0.6)
axes[1, 0].legend(loc="upper right")
# --- Hình 4: Trọng số của các đặc trưng (Feature Coefficients) ---
encoded_features = (
    binary_cols
    + ordinal_cols
    + list(
        clf_pipeline.named_steps["preprocessor"]
        .named_transformers_["nom"]
        .get_feature_names_out(nominal_cols)
    )
    + numeric_cols
)
coefs = clf_pipeline.named_steps["classifier"].coef_[0]
coef_df = pd.Series(coefs, index=encoded_features).sort_values()

coef_df.plot(
    kind="barh",
    ax=axes[1, 1],
    color=np.where(coef_df > 0, "#264653", "#E63946"),
    edgecolor="black",
)
axes[1, 1].set_title(
    "4. Tác động của các đặc trưng (Logistic Coefficients)",
    fontsize=11,
    fontweight="bold",
)
axes[1, 1].set_xlabel("Hệ số trọng số (Dương: Tăng nguy cơ, Âm: Giảm nguy cơ)")
axes[1, 1].grid(axis="x", linestyle="--", alpha=0.6)

# Căn chỉnh bố cục và lưu
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = os.path.join("plots", "kich_ban_1_ket_qua.png")
plt.savefig(plot_path, dpi=300)
plt.close()

# ==========================================
# 7. LƯU CHỈ SỐ RA FILE CSV TỔNG HỢP
# ==========================================
metrics_summary = pd.DataFrame(
    [
        {
            "Kịch bản": "KB 1 (Chỉ dùng Bảng)",
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
metrics_path = os.path.join("plots", "kich_ban_1_metrics.csv")
metrics_summary.to_csv(metrics_path, index=False)

print(f"\n=> ĐÃ XUẤT THÀNH CÔNG HÌNH VẼ: {plot_path}")
print(f"=> ĐÃ LƯU BẢNG ĐIỂM SỐ RA CSV  : {metrics_path}")