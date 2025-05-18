#benchmarking singleR
import scanpy as sc
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

h5ad_pred_path = "D:/Users/Xin/Downloads/CMML_mini2/newZheng68KtestsingleR.h5ad"
h5ad_gt_path = "D:/Users/Xin/Downloads/scBERT-master/scBERT-master_remote/data/Zheng68K_test_10pct.h5ad"
output_path = "D:/Users/Xin/Downloads/scBERT-master/singleR_result/processed_Zheng68KtestsingleR.h5ad"

adata = sc.read_h5ad(h5ad_pred_path)
adata_gt = sc.read_h5ad(h5ad_gt_path)

df_gt = adata_gt.obs[["celltype"]]
df_gt.index.name = "cell_id"
adata.obs = adata.obs.merge(df_gt, left_index=True, right_index=True, how="left")

gt_to_coarse = {
    'CD19+ B': 'B cells',
    'CD4+ T Helper2': 'CD4+ T cells',
    'CD4+/CD25 T Reg': 'CD4+ T cells',
    'CD4+/CD45RA+/CD25- Naive T': 'CD4+ T cells',
    'CD4+/CD45RO+ Memory': 'CD4+ T cells',
    'CD8+ Cytotoxic T': 'CD8+ T cells',
    'CD8+/CD45RA+ Naive Cytotoxic': 'CD8+ T cells',
    'CD14+ Monocyte': 'Monocytes',
    'CD56+ NK': 'NK cells',
    'Dendritic': 'Dendritic cells',
    'CD34+': 'Progenitors'
}

coarse_mapping = {
    'T_cells': 'CD8+ T cells',
    'T_cel': 'CD8+ T cells',
    'B_cell': 'B cells',
    'Pre-B_cell_CD34-': 'B cells',
    'Pro-B_cell_CD34+': 'Progenitors',
    'CMP': 'Progenitors',
    'GMP': 'Progenitors',
    'MEP': 'Progenitors',
    'HSC_-G-CSF': 'Progenitors',
    'Platelets': 'Progenitors',
    'Myelocyte': 'Progenitors',
    'Macrophage': 'Monocytes',
    'Monocyte': 'Monocytes',
    'DC': 'Dendritic cells',
    'NK_cell': 'NK cells'
}

# === Step 5: 应用映射
pred_raw = adata.obs["SingleR.labels"]
adata.obs["predicted_coarse"] = pred_raw.map(coarse_mapping).fillna(pred_raw)  # fallback: 保留原始标签
adata.obs["true_coarse"] = adata.obs["celltype"].map(gt_to_coarse).fillna("Unmapped")

# === Step 6: 过滤掉 unmapped 样本
mask = (adata.obs["predicted_coarse"].notna()) & (adata.obs["true_coarse"] != "Unmapped")

# === Step 7: 分类评估 ===
y_true = adata.obs["true_coarse"][mask]
y_pred = adata.obs["predicted_coarse"][mask]

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np

# === 计算混淆矩阵并归一化 ===
labels = sorted(y_true.unique())
conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

# 计算每一行的比例
conf_mat_norm = conf_mat.astype(np.float64)
row_sums = conf_mat_norm.sum(axis=1, keepdims=True)
conf_mat_norm = np.divide(conf_mat_norm, row_sums, where=row_sums != 0)

# 转换为 DataFrame 以便绘图
conf_df = pd.DataFrame(conf_mat_norm, index=labels, columns=labels)

# === 计算总体 accuracy ===
overall_accuracy = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {overall_accuracy:.2%}")
# === 绘图 ===
plt.figure(figsize=(10, 8))
sns.heatmap(conf_df, annot=True, fmt=".2f", cmap="Blues", cbar_kws={"label": "Proportion"})
plt.title(f"Confusion Matrix for SingleR (Reverse Mapping)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()