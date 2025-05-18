#extract noisy training dataset from noisy label training dataset for further correction analysis
import scanpy as sc
import os

input_file = "D:/Users/Xin/Downloads/scBERT-master/data4train/Zheng68K_train_data_noisy_20pct.h5ad"
output_file = "D:/Users/Xin/Downloads/scBERT-master/data4train/noisy_label_only4test.h5ad"

adata = sc.read_h5ad(input_file)

if "is_corrupted" not in adata.obs.columns:
    raise ValueError("Missing required field: 'is_corrupted' in adata.obs")

adata_corrupted = adata[adata.obs["is_corrupted"] == True].copy()

adata_corrupted.write(output_file)
print(f"[INFO] Saved corrupted-only dataset to: {output_file}")
print(f"[INFO] Dataset size: {adata_corrupted.n_obs} cells, {adata_corrupted.n_vars} genes")