#benchmark for singleR
import scanpy as sc
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


h5ad_pred_path = "D:/Users/Xin/Downloads/CMML_mini2/data/SingleR_Zheng68Knew.h5ad"
h5ad_gt_path = "D:/Users/Xin/Downloads/scBERT-master/scBERT-master_remote/data/Zheng68K_test_10pct.h5ad"
output_path = "D:/Users/Xin/Downloads/scBERT-master/singleR_result/processed_Zheng68KtestsingleR.h5ad"

adata = sc.read_h5ad(h5ad_pred_path)
adata_gt = sc.read_h5ad(h5ad_gt_path)
print(adata.obs)
print(np.unique(adata.obs["SingleR_pred"]))
print(np.unique(adata_gt.obs["celltype"]))

df_gt = adata_gt.obs[["celltype"]]
df_gt.index.name = "cell_id"
adata.obs = adata.obs.merge(df_gt, left_index=True, right_index=True, how="left")



label_mapping = {
    'B cells': 'CD19+ B',
    'CD4+ T cells': 'CD4+/CD25 T Reg', 
    'CD8+ T cells': 'CD8+ Cytotoxic T',
    'Dendritic cells': 'Dendritic',
    'Monocytes': 'CD14+ Monocyte',
    'NK cells': 'CD56+ NK',
    'Progenitors': 'CD34+',
    'T cells': 'CD8+ Cytotoxic T' 
}

pred_raw = adata.obs["SingleR_pred"]
adata.obs["predicted_celltype"] = pred_raw.map(label_mapping).fillna(pred_raw)
adata.write(output_path)

unmapped = pred_raw[~pred_raw.isin(label_mapping.keys())].unique()
if len(unmapped) > 0:
    print(f"[INFO] 以下预测标签未映射（已原样保留）: {unmapped}")


y_true = adata.obs["celltype"]
y_pred = adata.obs["predicted_celltype"]

mask = y_true.notna() & y_pred.notna()

print("\n📊 Classification Report:")
print(classification_report(y_true[mask], y_pred[mask]))

df = pd.DataFrame({
    'true': y_true[mask],
    'pred': y_pred[mask]
})
df['correct'] = df['true'] == df['pred']
summary = df.groupby('true').agg(
    count=('true', 'count'),
    accuracy=('correct', 'mean')
).reset_index()

print("\n Per-celltype accuracy summary:")
print(summary)

summary.to_csv("singleR_accuracy_by_celltype.csv", index=False)


adata.write(output_path)
print(f"\n 保存映射后的 AnnData 到：{output_path}")
plt.figure(figsize=(10, 5))
sns.barplot(data=summary, x='true', y='count')
plt.xticks(rotation=90)
plt.title("Cell Type Counts")
plt.xlabel("Cell Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

label_counts = adata.obs["SingleR.labels"].value_counts().reset_index()
label_counts.columns = ['label', 'count']

print(label_counts)

plt.figure(figsize=(10, 5))
sns.barplot(data=label_counts, x='label', y='count', order=label_counts['label'])
plt.xticks(rotation=90)
plt.title("Cell Type Counts (Pre-mapping SingleR.labels)")
plt.xlabel("SingleR Predicted Label")
plt.ylabel("Cell Count")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
sns.barplot(data=summary, x='true', y='accuracy')
plt.xticks(rotation=90)
plt.title("Prediction Accuracy per Cell Type")
plt.xlabel("Cell Type")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()
