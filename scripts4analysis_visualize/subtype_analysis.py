#analysis subtpye cell annotation for different methods

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


adata = sc.read_h5ad("D:/Users/Xin/Downloads/scBERT-master/predicted_result4baseline/scgpt_best/prediction_result.h5ad")


df = pd.DataFrame({
    'true': adata.obs['celltype'],
    'pred': adata.obs['predicted_celltype']
})

df['correct'] = df['true'].astype(str) == df['pred'].astype(str)
from sklearn.metrics import confusion_matrix
import numpy as np

labels = sorted(df['true'].unique())

cm = confusion_matrix(df['true'], df['pred'], labels=labels, normalize='true')
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, xticklabels=labels, yticklabels=labels,
            cmap="Blues", annot=True, fmt=".2f", cbar=True)
plt.title("Confusion Matrix of Cell Type Predictions (scGPT)")
plt.xlabel("Predicted Cell Type")
plt.ylabel("True Cell Type")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


summary = df.groupby('true').agg(
    count=('true', 'count'),
    accuracy=('correct', 'mean')  # mean of True/False == accuracy
).reset_index()
print(summary)

plt.figure(figsize=(10, 5))
sns.barplot(data=summary, x='true', y='count')
plt.xticks(rotation=90)
plt.title("Cell Type Counts")
plt.xlabel("Cell Type")
plt.ylabel("Count")
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

import scanpy as sc
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

adata = sc.read_h5ad("D:/Users/Xin/Downloads/scBERT-master/data/processed_scgptumapv2.h5ad")


df = pd.DataFrame({
    'true': adata.obs['celltype'],
    'pred': adata.obs['predicted_celltype']
})


label_map = {
    "CD14+ Monocyte": "Monocyte",
    "CD19+ B": "B cells",
    "CD34+": "Progenitor",
    "CD4+ T Helper2": "Th2",
    "CD4+/CD25 T Reg": "Treg",
    "CD4+/CD45RA+/CD25- Naive T": "Naive CD4",
    "CD4+/CD45RO+ Memory": "Memory CD4",
    "CD56+ NK": "NK",
    "CD8+ Cytotoxic T": "CD8",
    "CD8+/CD45RA+ Naive Cytotoxic": "Naive CD8",
    "Dendritic": "Dendritic"
}
df['true_short'] = df['true'].map(label_map)
df['pred_short'] = df['pred'].map(label_map)


labels_short = sorted(df['true_short'].dropna().unique())
cm = confusion_matrix(df['true_short'], df['pred_short'], labels=labels_short, normalize='true')


cm_df = pd.DataFrame(cm, index=labels_short, columns=labels_short)
cm_df.index.name = "True"
cm_df.columns.name = "Predicted"
cm_df = cm_df.reset_index().melt(id_vars="True", var_name="Pred", value_name="Proportion")

cm_df.to_csv("D:/Users/Xin/Downloads/scBERT-master/scBERT-master_remote/data_scgpt.csv", index=False)