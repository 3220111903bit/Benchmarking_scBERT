library(Seurat)
library(SingleR)
library(SummarizedExperiment)
library(SeuratDisk)
library(Matrix)


h5ad_file <-'D:/Users/Xin/Downloads/scBERT-master/scBERT-master_remote/Zheng68K_test_10pct.h5ad'
Convert(h5ad_file, dest = "h5seurat", overwrite = TRUE)
seurat_obj <-LoadH5Seurat("D:/Users/Xin/Downloads/CMML_mini2/data/Zheng68K.h5seurat")
 


Idents(seurat_obj) <- "celltype"


set.seed(42)
all_cells <- colnames(seurat_obj)
train_idx <- sample(seq_along(all_cells), size = 0.8 * length(all_cells))
train_cells <- all_cells[train_idx]
test_cells <- setdiff(all_cells, train_cells)

ref_seurat <- subset(seurat_obj, cells = train_cells)
test_seurat <- subset(seurat_obj, cells = test_cells)

ref_mat <- GetAssayData(ref_seurat, slot = "data")
ref_labels <- ref_seurat$celltype
ref_sce <- SummarizedExperiment(list(logcounts = as.matrix(ref_mat)),
                                colData = data.frame(label = ref_labels))

test_mat <- GetAssayData(test_seurat, slot = "data")
test_sce <- SummarizedExperiment(list(logcounts = as.matrix(test_mat)))

pred <- SingleR(test = test_sce, ref = ref_sce, labels = ref_sce$label)


test_seurat$SingleR_pred <- pred$labels
print(test_seurat@assays)

test_seurat <- NormalizeData(test_seurat)


test_seurat <- FindVariableFeatures(test_seurat)

test_seurat <- ScaleData(test_seurat)

test_seurat <- RunPCA(test_seurat, npcs = 30)


test_seurat <- RunUMAP(test_seurat, dims = 1:30)
test_seurat <- ScaleData(test_seurat)
p1 <- DimPlot(test_seurat, reduction = "umap", group.by = "SingleR_pred", label = TRUE) +
  ggtitle("UMAP: SingleR Predicted Cell Types") +
  theme_minimal()

print(p1)

#test_seurat <- RunPCA(test_seurat, npcs = 30)
#test_seurat <- RunUMAP(test_seurat, dims = 1:30)


SaveH5Seurat(test_seurat, filename = "D:/Users/Xin/Downloads/CMML_mini2/data/SingleR_Zheng68Knew.h5seurat", overwrite = TRUE)

Convert("D:/Users/Xin/Downloads/CMML_mini2/data/SingleR_Zheng68Knew.h5seurat", dest = "h5ad", overwrite = TRUE)
seurat_obj <-LoadH5Seurat("D:/Users/Xin/Downloads/CMML_mini2/data/SingleR_Zheng68Knew.h5seurat")

library(caret)
library(dplyr)
library(MLmetrics)
test_seurat <- seurat_obj
true_labels <- test_seurat$celltype
pred_labels <- test_seurat$SingleR_pred

#mapping
shorten_celltype <- function(vec) {
  map <- c(
    "CD14+ Monocyte" = "Monocyte",
    "CD19+ B" = "B cells",
    "CD34+" = "Progenitor",
    "CD4+ T Helper2" = "Th2",
    "CD4+/CD25 T Reg" = "Treg",
    "CD4+/CD45RA+/CD25- Naive T" = "Naive CD4",
    "CD4+/CD45RO+ Memory" = "Memory CD4",
    "CD56+ NK" = "NK",
    "CD8+ Cytotoxic T" = "CD8",
    "CD8+/CD45RA+ Naive Cytotoxic" = "Naive CD8",
    "Dendritic" = "Dendritic"
  )
  return(unname(map[vec]))
}


true_short <- shorten_celltype(true_labels)
pred_short <- shorten_celltype(pred_labels)
df <- data.frame(True = true_short, Pred = pred_short) %>% filter(!is.na(True) & !is.na(Pred))


cell_count_order <- c(
  "CD8", "Naive CD8", "NK", "Treg", "B cells", 
  "Memory CD4", "Monocyte", "Naive CD4", "Dendritic", "Progenitor", "Th2"
)

df$True <- factor(df$True, levels = cell_count_order)
df$Pred <- factor(df$Pred, levels = cell_count_order)

cm <- table(df$True, df$Pred)
cm_norm <- prop.table(cm, margin = 1) 

cm_long <- as.data.frame(cm_norm)
colnames(cm_long) <- c("True", "Pred", "Proportion")

cm_long$True <- factor(cm_long$True, levels = rev(cell_count_order))  # 上到下
cm_long$Pred <- factor(cm_long$Pred, levels = cell_count_order) 
accuracy_per_class <- cm_long %>%
  filter(True == Pred) %>%
  arrange(desc(Proportion))  


print(accuracy_per_class)# 左到右

acc <- Accuracy(pred_labels, true_labels)
classes <- sort(unique(true_labels))


recall_vec <- c()
f1_vec <- c()


for (cls in classes) {
  
  y_true_bin <- as.integer(true_labels == cls)
  y_pred_bin <- as.integer(pred_labels == cls)
  
 
  rec <- tryCatch(Recall(y_pred_bin, y_true_bin, positive = 1), error = function(e) NA)
  f1 <- tryCatch(F1_Score(y_pred_bin, y_true_bin, positive = 1), error = function(e) NA)
  
  recall_vec <- c(recall_vec, rec)
  f1_vec <- c(f1_vec, f1)
}

recall_macro <- mean(recall_vec, na.rm = TRUE)
f1_macro <- mean(f1_vec, na.rm = TRUE)

cat("📊 Overall Evaluation Metrics:\n")
cat(sprintf("✅ Accuracy: %.3f\n", acc))
cat(sprintf("✅ Macro F1: %.3f\n", f1_macro))
cat(sprintf("✅ Macro Recall: %.3f\n", recall_macro))

eval_df <- data.frame(
  true = true_labels,
  pred = pred_labels
)


celltype_metrics <- eval_df %>%
  group_by(true) %>%
  summarise(
    count = n(),
    accuracy = mean(pred == true),
    recall = sum(pred == true) / sum(true == true),
    f1 = ifelse(sum(pred == true) + sum(true == true) == 0, NA,
                2 * sum(pred == true & true == true) / 
                  (sum(pred == true) + sum(true == true)))
  ) %>%
  arrange(desc(count))

cat("\n📊 Per-celltype Evaluation Summary:\n")
print(celltype_metrics)

# 保存为 CSV
write.csv(celltype_metrics, "Zheng68K_SingleR_per_celltype_metrics.csv", row.names = FALSE)
