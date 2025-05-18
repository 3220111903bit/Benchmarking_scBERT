import anndata

def check_reference_file(adata: anndata.AnnData) -> dict:
    
    label_keys = ['celltype', 'cell_type', 'Celltype', 'label']
    found_keys = [key for key in label_keys if key in adata.obs.columns]
    return {
        "type": "reference",
        "success": len(found_keys) > 0,
        "found_labels": found_keys,
        "message": f"Found label columns: {found_keys}" if found_keys else "No label column found in .obs"
    }

def check_test_file(adata: anndata.AnnData) -> dict:
    
    has_umap = 'X_umap' in adata.obsm
    return {
        "type": "test",
        "success": has_umap,
        "message": "'X_umap' found in .obsm" if has_umap else "'X_UMAP' not found in .obsm"
    }

def check_scgpt_compatibility(file_path: str, is_reference: bool = True) -> dict:
    
    try:
        adata = anndata.read_h5ad(file_path)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to load file: {file_path}"
        }

    return check_reference_file(adata) if is_reference else check_test_file(adata)

#print(check_scgpt_compatibility("D:/Users/Xin/Downloads/scBERT-master/data_used/Zheng68K_train_90pct.h5ad", is_reference=True))
#print(check_scgpt_compatibility("D:/Users/Xin/Downloads/scBERT-master/data_used/Zheng68K_test4scgpt.h5ad", is_reference=False))

import scanpy as sc
adata = sc.read_h5ad("D:/Users/Xin/Downloads/scBERT-master/data/processed_scgptumap.h5ad")


adata_tmp = adata.copy()
sc.pp.normalize_total(adata_tmp)
sc.pp.log1p(adata_tmp)
sc.pp.highly_variable_genes(adata_tmp, n_top_genes=2000, subset=True)
sc.pp.scale(adata_tmp)
sc.tl.pca(adata_tmp, svd_solver='arpack')
sc.pp.neighbors(adata_tmp)
sc.tl.umap(adata_tmp)


adata.obsm["X_umap"] = adata_tmp.obsm["X_umap"]
sc.pl.umap(adata)
#adata.write("D:/Users/Xin/Downloads/scBERT-master/data_used/Zheng68K_test_4scgpt.h5ad")

adata.var["gene_name"] = adata.var.index.astype(str)

adata.write("D:/Users/Xin/Downloads/scBERT-master/data/processed_scgptumapv2.h5ad")
""""""