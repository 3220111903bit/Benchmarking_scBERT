#tqdm and evaluation for metrics are further integrated by Xin
import argparse
import os
import numpy as np
#import scanpy as sc
import torch
import pickle as pkl
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm  # 加载进度条
import pandas as pd
from performer_pytorch import PerformerLM
import anndata as ad

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--label_dict_dir", type=str, default='./labeldict', help='Label dictionary directory path.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--novel_type", type=bool, default=False, help='Novel cell type exists or not.')
parser.add_argument("--unassign_thres", type=float, default=0.5, help='Threshold for assigning Unassigned label.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad', help='Path to input .h5ad file.')
parser.add_argument("--model_path", type=str, default='./finetuned.pth', help='Path to finetuned model.')
parser.add_argument("--output_dir", type=str, default='./prediction_outputs', help='Directory to save predictions and results.')
args = parser.parse_args()

# Setup
SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num + 2
UNASSIGN = args.novel_type
label_dict_dir = args.label_dict_dir
UNASSIGN_THRES = args.unassign_thres if UNASSIGN else 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(args.output_dir, exist_ok=True)

# Define Identity Head
class Identity(torch.nn.Module):
    def __init__(self, dropout=0., h_dim=100, out_dim=10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (1, 200))
        self.act = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(SEQ_LEN, 512)
        self.act1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(512, h_dim)
        self.act2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout)
        self.fc3 = torch.nn.Linear(h_dim, out_dim)

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Load data
adata = ad.read_h5ad(args.data_path)
X = adata.X
print(f"[INFO] Loaded data shape: {X.shape}")

# Load label_dict
with open(label_dict_dir, "rb") as fp:
    label_dict = pkl.load(fp)
reverse_label_dict = {i: label for i, label in enumerate(label_dict)}

# Load model
model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=args.pos_embed
)
model.to_out = Identity(dropout=0., h_dim=128, out_dim=len(label_dict))

ckpt = torch.load(args.model_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
model = model.to(device)

# Predict
pred_finals = []
novel_indices = []

with torch.no_grad():
    for idx in tqdm(range(X.shape[0]), desc="Predicting"):
        full_seq = X[idx].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        full_seq = full_seq.unsqueeze(0)

        logits = model(full_seq)
        prob = torch.nn.functional.softmax(logits, dim=-1)
        pred_class = prob.argmax(dim=-1).item()

        if torch.max(prob).item() < UNASSIGN_THRES:
            novel_indices.append(idx)
        pred_finals.append(pred_class)

# Decode predictions
pred_labels = [reverse_label_dict[p] for p in pred_finals]
for i in novel_indices:
    pred_labels[i] = "Unassigned"

# Save results to AnnData and CSV
adata.obs["predicted_celltype"] = pred_labels
adata.write(os.path.join(args.output_dir, "prediction_result.h5ad"))
print(f"[INFO] Saved annotated AnnData to {args.output_dir}/prediction_result.h5ad")

df = pd.DataFrame({
    "cell_id": adata.obs_names,
    "pred_label": pred_labels
})
df.to_csv(os.path.join(args.output_dir, "prediction_result.csv"), index=False)
print(f"[INFO] Saved CSV predictions to {args.output_dir}/prediction_result.csv")

# Aditional: Evaluation
# Evaluate if GT available
eval_file = os.path.join(args.output_dir, "evaluation_metrics.txt")
if "celltype" in adata.obs.columns:
    y_true = np.array(adata.obs["celltype"])
    y_pred = np.array(pred_labels)

    mask = y_pred != "Unassigned"
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    acc = accuracy_score(y_true_masked, y_pred_masked)
    f1 = f1_score(y_true_masked, y_pred_masked, average='macro')
    recall = recall_score(y_true_masked, y_pred_masked, average='macro')

    print(f"[Eval] Accuracy: {acc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")
    with open(eval_file, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nF1 Score: {f1:.4f}\nRecall: {recall:.4f}\n")
    print(f"[INFO] Evaluation saved to {eval_file}")
else:
    print("[Warning] No ground truth label (obs['celltype']) found. Skipping evaluation.")
