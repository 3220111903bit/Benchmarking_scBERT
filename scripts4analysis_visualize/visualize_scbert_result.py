#try visualizing scbert result 
import scanpy as sc
import matplotlib.pyplot as plt
import os

input_h5ad ="D:/Users/Xin/Downloads/scBERT-master/data/preprocessed_Zheng68K_test_10pct.h5ad"
output_dir = "D:/Users/Xin/Downloads/scBERT-master/scBERT-master_remote/prediction_outputs/figures"
fig_dir = os.path.join(output_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

adata = sc.read_h5ad(input_h5ad)

print("[INFO] Preprocessing: normalization, HVGs, scaling, PCA...")

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')

sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata)


umap_pred_path = os.path.join(fig_dir, "umap_predicted.png")
umap_gt_path = os.path.join(fig_dir, "umap_gt.png")

sc.pl.umap(
    adata,
    color="predicted_celltype",
    title="UMAP: Predicted Cell Types",
    save=False,
    show=False
)
plt.savefig(umap_pred_path, dpi=300)
print(f"[INFO] Saved UMAP (predicted) to: {umap_pred_path}")

if "celltype" in adata.obs.columns:
    sc.pl.umap(
        adata,
        color="celltype",
        title="UMAP: Ground Truth Cell Types",
        save=False,
        show=False
    )
    plt.savefig(umap_gt_path, dpi=300)
    print(f"[INFO] Saved UMAP (ground truth) to: {umap_gt_path}")

adata.write(os.path.join(output_dir, "prediction_with_umap.h5ad"))
print(f"[INFO] Saved updated AnnData to: {output_dir}/prediction_with_umap.h5ad")