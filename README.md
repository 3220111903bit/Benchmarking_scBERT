# Benchmarking_scBERT

A repository demonstrating the reusability and benchmarking of **scBERT** on various single-cell annotation tasks. This version includes recent **overwrites to the fine-tune and prediction pipelines**, expanded **dependency checks**, **TensorBoard logging**, and improved **visualization scripts** for downstream analysis.

> 🔬 Originally based on: [TencentAILabHealthcare/scBERT](https://github.com/TencentAILabHealthcare/scBERT)

---

## ✨ Project Updates

- ✅ Rewritten `finetune_updatev4.py` and `predict_updatedv1.py` with cleaner argument control and better logging.
- 📦 Checked and updated dependencies in `requirements_update.txt` (e.g., TensorBoard, seaborn, sklearn).
- 📊 Integrated **TensorBoard** for loss/accuracy tracking during training.
- 🎨 Added `scripts4analysis_visualize/` folder for confusion matrix, subtype analysis, and barplot summaries.

---

## 🔧 Installation

```bash
# Clone the repo
git clone https://github.com/YourUsername/Benchmarking_scBERT.git
cd Benchmarking_scBERT

# Install dependencies (tested on Python 3.8.1)
pip install -r requirements_update.txt
```

You will also need the following files placed in appropriate folders (or update paths in scripts accordingly):
- `panglao_pretrain.pth` → `./weight/`
- `Zheng68K.h5ad` or your own `.h5ad` file → `./setting_data/`

---
## 📁 Required Files

To run fine-tuning and evaluation, please make sure the following files are present in the correct directories:

./weight/
├── panglao_pretrain.pth

./setting_data/
├── Zheng68K.h5ad
├── panglao_1000.h5ad

kotlin
Copy
Edit

These files are **not included** in this repository due to size or licensing restrictions.

To obtain the files, please contact:

📧 **fionafyang@tencent.com**


## 🚀 Usage

### 🔹 Fine-tuning the model

```bash
python finetune_updatev4.py \
  --data_path "./setting_data/your_train_data.h5ad" \
  --model_path "./weight/panglao_pretrain.pth" \
  --log_dir "./logs" \
  --epochs 10 \
```

TensorBoard logs will be saved to the specified `log_dir`. You can view them with:
```bash
tensorboard --logdir ./logs
```

---

### 🔹 Running prediction

```bash
python predict_updatedv1.py \
  --data_path "./setting_data/your_test_data.h5ad" \
  --model_path "./weight/finetuned_model.pth" \
  --label_dict_dir "./label_dict" \
```

This will output predicted labels in `.h5ad` format under `prediction_outputs/`.

---

## 📊 Visualization & Analysis

You can explore and visualize results with:

```bash
# Custom cell-type-wise accuracy and plots
python scripts4analysis_visualize/multiumap.py

# Summarize attention scores (optional)
python attn_sum_save.py
```

These scripts support:
- Accuracy per cell type
- Cell type bar plots (with normalization)
- Confusion matrix heatmaps
- Subtype annotation breakdowns

You may also customize your own plots with the provided templates.

---

## 📁 Folder Structure

```
Benchmarking_scBERT/
├── performer_pytorch/           # Performer-based model
├── prediction_outputs/         # Saved prediction results (.h5ad)
├── scripts4analysis_visualize/ # Visualization and analysis scripts
├── setting_data/               # Preprocessed input files
├── weight/                     # Pretrained and finetuned weights
├── preprocess.py               # Data normalization & formatting
├── finetune_updatev4.py        # Updated training script
├── predict_updatedv1.py        # Updated prediction script
├── utils.py                    # Utility functions
├── requirements_update.txt     # Python dependencies
├── attn_sum_save.py            # Optional attention score analysis
└── README.md
```

---

## 📜 License

This project follows the LICENSE provided in the original scBERT repository.
