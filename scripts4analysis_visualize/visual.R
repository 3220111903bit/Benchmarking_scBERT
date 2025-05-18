library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(RColorBrewer)

#visual1: cell distribution

df <- data.frame(
  celltype = c(
    "Monocyte", "B cells", "Progenitor", "Th2", "Treg",
    "Naive CD4", "Memory CD4", "NK", "CD8", "Naive CD8", "Dendritic"
  ),
  count = c(274, 587, 20, 13, 650, 210, 300, 817, 2104, 1667, 203)
)


df <- df %>%
  mutate(percent = round(100 * count / sum(count), 2)) %>%
  arrange(count) %>%
  mutate(celltype = factor(celltype, levels = celltype))  # 保证绘图排序


ggplot(df, aes(x = celltype, y = count)) +
  geom_bar(stat = "identity", fill = "#2c3e50") +
  geom_text(aes(label = paste0(count, " (", percent, "%)")),
            hjust = -0.1, size = 3.5) +
  coord_flip() +
  theme_classic(base_size = 16) +
  labs(title = "Cell Type Distribution in Zheng68K",
       x = "Cell Type", y = "Number of Cells") +
  ylim(0, max(df$count) * 1.2) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(color = "grey80", linetype = "dashed"),
        plot.title = element_text(face = "bold"))


# visual2: metrics for baseline
df <- data.frame(
  Experiment = c("Baseline", "Baseline", "Baseline"),
  Model = c("scGPT", "scBERT", "SingleR"),
  Acc = c(0.748, 0.7582, 0.369),
  F1 = c(0.629, 0.6518, 0.445),
  Recall = c(0.605, 0.6408, 0.617)
)


df_long <- df %>%
  select(Model, Acc, F1, Recall) %>%
  pivot_longer(cols = c("Acc", "F1", "Recall"), names_to = "Metric", values_to = "Score")


ggplot(df_long, aes(x = Model, y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(aes(label = sprintf("%.2f", Score)), 
            position = position_dodge(width = 0.7), vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c("#4D4D4D", "#1B9E77", "#D95F02")) +
  theme_classic(base_size = 16) +
  labs(x = "Model", y = "Score", fill = "Metric") +
  ylim(0, 0.85) +
  theme(panel.grid.major.x = element_blank(),
        axis.title = element_text(face = "bold"),
        plot.title = element_text(face = "bold"))


shorten_celltype <- function(celltype_vec) {
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
  return(map[celltype_vec])
}


get_confusion_matrix_df <- function(true_labels, pred_labels) {
  df <- data.frame(true = shorten_celltype(true_labels),
                   pred = shorten_celltype(pred_labels))
  
  
  labels <- sort(unique(c(df$true, df$pred)))
  df$true <- factor(df$true, levels = labels)
  df$pred <- factor(df$pred, levels = labels)

  cm_table <- table(df$true, df$pred)
  cm_prop <- prop.table(cm_table, 1)
  
  
  cm_df <- as.data.frame(cm_prop)
  colnames(cm_df) <- c("True", "Pred", "Proportion")
  return(cm_df)
}


cell_count_order <- c(
  "CD8", "Naive CD8", "NK", "Treg", "B cells", 
  "Memory CD4", "Monocyte", "Naive CD4", "Dendritic", "Progenitor", "Th2"
)

cm_df <- read.csv("D:/Users/Xin/Downloads/scBERT-master/scBERT-master_remote/data_scGPT.csv")
cm_df$True <- factor(cm_df$True, levels = rev(cell_count_order))
cm_df$Pred <- factor(cm_df$Pred, levels = cell_count_order)


ggplot(cm_df, aes(x = Pred, y = True, fill = Proportion)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", Proportion)), size = 3) +
  scale_fill_gradientn(colors = brewer.pal(9, "Blues"), limits = c(0, 1)) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 10),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
  ) +
  labs(
    title = "SingleR",
    x = "Predicted Cell Type",
    y = "True Cell Type",
    fill = "Proportion"
  )


