# 🧠 DS542 Deep Learning for Data Science — Midterm Challenge (Spring 2025)

### 📝 Author: Viktoria Zruttova  
📅 Date: March 27, 2025  
🏆 Kaggle Username: *Viktoria Zruttova*  
📊 Best Kaggle Score: **0.57876** (Part 3)

---

## 📌 Overview

This repository contains my submission for the DS542 Midterm Challenge, which focuses on image classification using the CIFAR-100 dataset. The challenge was divided into three parts, each increasing in complexity:

1. **Part 1 – Simple CNN:** Design and train a baseline convolutional neural network from scratch.
2. **Part 2 – More Sophisticated CNN (ResNet18):** Train a deeper model and introduce basic regularization and augmentation.
3. **Part 3 – Transfer Learning (ResNet50):** Fine-tune a pretrained model using advanced training techniques and optimization strategies.

---

## 🤖 AI Disclosure

I used the free version of **ChatGPT** for assistance throughout this project. The AI helped me understand CNN architectures, implement regularization techniques, and debug training loops. I personally wrote the initial model definitions, training logic, and hyperparameter tuning strategies, while AI support was primarily used for learning, validation, and code refinement. All code includes detailed comments describing the functionality.

---

## 📦 Project Structure

```bash
.
├── part1_simple_cnn.py         # Code for baseline CNN model
├── part2_resnet18.py           # Code for training ResNet18 from scratch
├── part3_resnet50_transfer.py  # Code for fine-tuning pretrained ResNet50
├── utils/                      # Helper functions (data loaders, transforms, etc.)
├── requirements.txt
└── README.md                   # This file

---

## 🧠 Model Summary

### 🔹 Part 1: Simple CNN  
- **Architecture:** 3 Conv layers + FC layer  
- **Optimizer:** SGD with momentum  
- **Scheduler:** StepLR  
- **Regularization:** None  
- **Kaggle Score:** **0.16158**

### 🔹 Part 2: ResNet18  
- **Modifications:** Custom classifier head with dropout  
- **Data Augmentation:** Horizontal flip, crop, color jitter  
- **Optimizer:** AdamW + CosineAnnealingLR  
- **Regularization:** Dropout (p=0.5), weight decay (1e-4)  
- **Kaggle Score:** **0.30093**

### 🔹 Part 3: ResNet50 (Pretrained)  
- **Transfer Learning:** Fine-tuned last 50 parameters  
- **Data Augmentation:** Strong augmentation + Mixup  
- **Regularization:** Dropout, BatchNorm, Label Smoothing (0.1), Early Stopping  
- **Optimizer:** AdamW  
- **Scheduler:** Warmup + CosineAnnealingWarmRestarts  
- **Kaggle Score:** **0.57876**

---

## ⚙️ Hyperparameter Tuning

| Parameter         | Value       | Reasoning |
|------------------|-------------|-----------|
| Batch Size       | 32          | Balanced compute and performance |
| Learning Rate    | 0.001       | Empirically best after testing |
| Epochs           | 200         | Early stopping applied (patience=40) |
| Optimizer        | AdamW       | Combines adaptive learning with regularization |
| Scheduler        | Warmup + Cosine | Helps escape local minima and stabilize learning |

---

## 🧪 Regularization & Augmentation Techniques

- **Dropout (0.5):** Reduces overfitting in FC layers  
- **Weight Decay (1e-4):** Penalizes large weights  
- **Early Stopping:** Monitors validation loss to prevent overfitting  
- **Label Smoothing (0.1):** Reduces overconfidence in predictions  
- **Gradient Clipping (max norm=1.0):** Prevents exploding gradients  
- **Data Augmentation:**  
  - Basic: Cropping, flipping, color jitter  
  - Advanced: Random erasing, Mixup blending for improved robustness

---

## 📊 Experiment Tracking (W&B)

All experiments were tracked using **Weights & Biases**, including:
- Training & validation accuracy/loss
- Learning rate trends
- Early stopping behavior
- Model checkpoints  
These visualizations helped identify overfitting and monitor performance improvements across training iterations.

---

## 📈 Results

| Part | Model         | Kaggle Score |
|------|---------------|--------------|
| 1    | Simple CNN    | 0.16158      |
| 2    | ResNet18      | 0.30093      |
| 3    | ResNet50 (TL) | 0.57876      |

Part 3 achieved the best performance due to its deep architecture, transfer learning, strong regularization, and carefully tuned augmentation strategies.

---

## 🚀 How to Run

\`\`\`bash
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run each model script independently
python part1_simple_cnn.py
python part2_resnet18.py
python part3_resnet50_transfer.py
\`\`\`

---

## 📬 Acknowledgments

Thanks to the DS542 instructors for the structured challenge and to **Weights & Biases** for the free experiment tracking tools.  
Also, a shoutout to ChatGPT for debugging help and support in exploring deep learning concepts!
