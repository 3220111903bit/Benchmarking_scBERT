#generate noisy labels
import scanpy as sc
import numpy as np
import random
import os

def corrupt_labels(adata, label_key="celltype", corruption_rate=0.2, seed=42):
    
    np.random.seed(seed)
    random.seed(seed)
    
    labels = adata.obs[label_key].to_numpy()
    unique_labels = np.unique(labels)
    
    n_total = len(labels)
    n_corrupt = int(corruption_rate * n_total)
    
    corrupt_indices = np.random.choice(n_total, n_corrupt, replace=False)
    noisy_labels = labels.copy()
    is_corrupted = np.full(n_total, False)

    for idx in corrupt_indices:
        original_label = labels[idx]
        other_labels = [l for l in unique_labels if l != original_label]
        noisy_labels[idx] = random.choice(other_labels)
        is_corrupted[idx] = True

    # 添加到 AnnData.obs
    adata.obs["label_true"] = labels
    adata.obs["label_noisy"] = noisy_labels
    adata.obs["is_corrupted"] = is_corrupted

    print(f"[INFO] Corrupted {n_corrupt}/{n_total} labels (~{corruption_rate*100:.0f}%).")
    return adata

if __name__ == "__main__":
    # ==== 配置 ====
    input_path = "D:/Users/Xin/Downloads/scBERT-master/data4train/preprocessed_Zheng68K_train_90pct.h5ad"        # 原始干净训练集
    output_path = "D:/Users/Xin/Downloads/scBERT-master/data4train/train_data_noisy_20pct.h5ad"  # 打乱后的输出路径
    label_key = "celltype"
    corruption_rate = 0.2
    seed = 42

    # ==== 主流程 ====
    print(f"[INFO] Loading data from: {input_path}")
    adata = sc.read_h5ad(input_path)

    adata = corrupt_labels(
        adata,
        label_key=label_key,
        corruption_rate=corruption_rate,
        seed=seed
    )

    adata.write(output_path)
    print(f"[INFO] Saved noisy dataset to: {output_path}")
