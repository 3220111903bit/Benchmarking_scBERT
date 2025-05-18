library(SingleR)
library(celldex)
library(Seurat)
library(SeuratObject)
library(SummarizedExperiment)
library(scater)
library(SeuratDisk)
library(ggplot2)
h5ad_file <-"a"
Convert(h5ad_file, dest = "h5seurat", overwrite = TRUE)
seurat_obj <- LoadH5Seurat("D:/Users/Xin/Downloads/scBERT-master/data/Zheng68K_test_10pct.h5seurat", meta.data = FALSE, misc = FALSE)


table(seurat_obj$celltype)
length(unique(seurat_obj$celltype))


df <- as.data.frame(table(seurat_obj$celltype))
colnames(df) <- c("CellType", "Count")

ggplot(df, aes(x = reorder(CellType, -Count), y = Count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_classic() +
  labs(x = "Cell Type", y = "Number of Cells", title = "Cell Type Distribution") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



