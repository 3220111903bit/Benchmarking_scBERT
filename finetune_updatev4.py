# -*- coding: utf-8 -*-
#tensorboard, tqdm and StratifiedShuffleSplit are integrated into the script by Xin
import os
import gc
import argparse
import json
import random
import math
import random
import sys
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from performer_pytorch import PerformerLM
import anndata as ad
from utils import *
import pickle as pkl
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=10, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad', help='Path of data for finetune.')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth', help='Path of pretrained model.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='finetune', help='Finetuned model name.')
args = parser.parse_args()

rank = int(os.environ.get("RANK", 0))
local_rank = args.local_rank
is_master = local_rank == 0
SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every
PATIENCE = 10
UNASSIGN_THRES = 0.0
CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed
model_name = args.model_name
ckpt_dir = args.ckpt_dir

# TensorBoard writer
global_step = 0
if is_master:
    writer = SummaryWriter(log_dir=f"./logs/tensorboard/{model_name}")
    print(f"[TensorBoard] Initialized writer at ./logs/tensorboard/{model_name}")
    


dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()
seed_all(SEED + torch.distributed.get_rank())

class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]

class Identity(nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
def get_reduced(val, local_rank, dst, world_size):
    val_tensor = torch.tensor([val], dtype=torch.float32).to(local_rank)
    dist.reduce(val_tensor, dst=dst, op=dist.ReduceOp.SUM)
    return (val_tensor.item() / world_size) if local_rank == dst else None

def distributed_concat(tensor, total_size, world_size):
    output_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:total_size]
# load data
data = ad.read_h5ad(args.data_path)
print('data_loaded')
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)
with open('label_dict', 'wb') as fp:
    pkl.dump(label_dict, fp)
with open('label', 'wb') as fp:
    pkl.dump(label, fp)
class_num = np.unique(label, return_counts=True)[1].tolist()
class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
label = torch.from_numpy(label)
data = data.X

acc = []
f1 = []
f1w = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pred_list = pd.Series(['un'] * data.shape[0])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for index_train, index_val in sss.split(data, label):
    data_train, label_train = data[index_train], label[index_train]
    data_val, label_val = data[index_val], label[index_val]
    train_dataset = SCDataset(data_train, label_train)
    val_dataset = SCDataset(data_val, label_val)

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=0)
print('dataloaded')

model = PerformerLM(num_tokens = CLASS, dim = 200, depth = 6, max_seq_len = SEQ_LEN, heads = 10, local_attn_heads = 0, g2v_position_emb = POS_EMBED_USING)
ckpt = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
for param in model.parameters():
    param.requires_grad = False
for param in model.norm.parameters():
    param.requires_grad = True
for param in model.performer.net.layers[-2].parameters():
    param.requires_grad = True
model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=15, cycle_mult=2, max_lr=LEARNING_RATE, min_lr=1e-6, warmup_steps=5, gamma=0.9)
loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)

dist.barrier()
log_records = []
trigger_times = 0
max_acc = 0.0

for i in range(1, EPOCHS+1):
    train_loader.sampler.set_epoch(i)
    model.train()
    dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {i}/{EPOCHS}", file=sys.stdout) if is_master else enumerate(train_loader)

    for index, (data, labels) in loop:
        data, labels = data.to(device), labels.to(device)
        if index % GRADIENT_ACCUMULATION != 0:
            with model.no_sync():
                logits = model(data)
                loss = loss_fn(logits, labels)
                loss.backward()
        else:
            logits = model(data)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
            optimizer.step()
            optimizer.zero_grad()
            if is_master:
                correct_num = torch.eq(logits.argmax(dim=-1), labels).sum().item()
                acc = correct_num / labels.size(0)
                writer.add_scalar("Train/loss", loss.item(), global_step)
                writer.add_scalar("Train/acc", acc, global_step)
                print(f"[TensorBoard] step={global_step}, loss={loss.item():.4f}, acc={acc:.4f}")
                global_step += 1

        running_loss += loss.item()
        cum_acc += torch.eq(logits.argmax(dim=-1), labels).sum().item() / labels.size(0)

    scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * cum_acc / len(train_loader)

    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)

    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')

    # ===== Validation =====
    if i % VALIDATE_EVERY == 0:
        model.eval()
        dist.barrier()
        running_loss = 0.0
        predictions = []
        truths = []

        with torch.no_grad():
            for index, (data_v, labels_v) in enumerate(val_loader):
                data_v, labels_v = data_v.to(device), labels_v.to(device)
                logits = model(data_v)
                loss = loss_fn(logits, labels_v)
                running_loss += loss.item()

                softmax = nn.Softmax(dim=-1)
                final_prob = softmax(logits)
                final = final_prob.argmax(dim=-1)
                final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                predictions.append(final)
                truths.append(labels_v)

            predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
            truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)

            no_drop = predictions != -1
            predictions = np.array((predictions[no_drop]).cpu())
            truths = np.array((truths[no_drop]).cpu())

            cur_acc = accuracy_score(truths, predictions)
            f1 = f1_score(truths, predictions, average='macro')
            val_loss = running_loss / (index + 1)
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)

            if is_master:
                print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | F1 Score: {f1:.6f}  ==')
                print(confusion_matrix(truths, predictions))
                print(classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4))
                writer.add_scalar("Val/acc", cur_acc, global_step)
                writer.add_scalar("Val/f1", f1, global_step)
                writer.add_scalar("Val/loss", val_loss, global_step)
                log_records.append(f"[Epoch {i} | Acc={cur_acc:.4f} | F1={f1:.4f} | Loss={val_loss:.4f}]")

                # ========== Save per-epoch model ==========
                os.makedirs(ckpt_dir, exist_ok=True)
                save_path_epoch = os.path.join(ckpt_dir, f"{model_name}_epoch_{i}.pth")
                torch.save({
                    'epoch': i,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': val_loss
                }, save_path_epoch)
                print(f"[Checkpoint] Epoch {i} model saved to {save_path_epoch}")

                # ========== Save best model if improved ==========
                if cur_acc > max_acc:
                    max_acc = cur_acc
                    trigger_times = 0
                    save_best = os.path.join(ckpt_dir, f"{model_name}_best.pth")
                    torch.save({
                        'epoch': i,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': val_loss
                    }, save_best)
                    print(f"[Checkpoint] Best model updated at epoch {i}, saved to {save_best}")
                else:
                    trigger_times += 1
                    if trigger_times > PATIENCE:
                        print(f"[Early Stopping] Triggered at epoch {i}")
                        break

        del predictions, truths

if is_master:
    writer.close()
    with open('./logs/final_summary.txt', 'a') as f:
        f.write("\n".join(log_records) + "\n")
