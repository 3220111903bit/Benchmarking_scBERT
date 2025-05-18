#Umap generation for the dataset and visualization for results
import scanpy as sc
import matplotlib.pyplot as plt

paths = {
    "scBERT": "D:/Users/Xin/Downloads/scBERT-master/predicted_result4baseline/scgpt_best/prediction_result.h5ad",
    "GT": "D:/Users/Xin/Downloads/scBERT-master/data/preprocessed_Zheng68K_test_10pct.h5ad",
    "SingleR": "D:/Users/Xin/Downloads/CMML_mini2/data/SingleR_Zheng68Knew.h5ad",
    "scGPT": "D:/Users/Xin/Downloads/scBERT-master/data/processed_scgptumapv2.h5ad"
}

adatas = {name: sc.read_h5ad(path) for name, path in paths.items()}

for name, adata in adatas.items():
    if name == "GT":
        adata.obs["annotation"] = adata.obs["celltype"]
    elif name == "SingleR":

        adata.obs["annotation"] = adata.obs["SingleR_pred"]
    else:
        adata.obs["annotation"] = adata.obs["predicted_celltype"]

    adata.obs["annotation"] = adata.obs["annotation"].astype("category")

def get_color_map(adata, key='annotation'):
    unique_types = adata.obs[key].cat.categories
    palette = sc.pl.palettes.default_102[:len(unique_types)]
    return dict(zip(sorted(unique_types), palette))

color_map = get_color_map(adatas["scBERT"], key="annotation")

def preprocess_adata(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)


for name, adata in adatas.items():
    print(f"🔄 Processing {name}...")

    if "X_umap" not in adata.obsm:
        preprocess_adata(adata)

    palette = [color_map.get(ct, "#CCCCCC") for ct in adata.obs["annotation"].cat.categories]

    sc.pl.umap(
        adata,
        color="annotation",
        palette=palette,
        title=name,
        show=True
    )
