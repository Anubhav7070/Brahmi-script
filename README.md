# 🪨 Vatteluttu Script Recognition using Hybrid Deep Learning Models

## 📌 Overview
This project implements a **comprehensive deep learning benchmark** for recognizing **Vatteluttu script characters** from **8th-century Tamil stone inscriptions**.  
It evaluates classical CNNs, sequence models, transfer learning architectures, and **hybrid Vision Transformer–based models** to study performance on low-resource ancient scripts.

The work is aimed at **digital epigraphy, historical document analysis, and heritage AI research**.

---

## 📂 Dataset
- **Source:** siddharthadevanv/8th-century-tamil-inscriptions (Kaggle)
- **Script:** Vatteluttu (ancient Tamil script)
- **Data Type:** Categorized character image dataset
- **Preprocessing Steps:**
  - Resize images to `128 × 128`
  - RGB normalization
  - Data augmentation:
    - Rotation
    - Width & height shift
    - Zoom & shear
    - Horizontal flip

---

## 🧠 Models Implemented
- **CNN** (custom convolutional network)
- **BiLSTM** (sequence modeling from image rows)
- **Transfer Learning Models**
  - VGG16
  - Xception
  - InceptionV3
- **Hybrid Architectures**
  - VGG16 + InceptionV3 + Vision Transformer block
  - VGG16 + Xception + Vision Transformer block

---

## ⚙️ Training Configuration
- **Framework:** TensorFlow / Keras
- **Optimizers:** Adam, RMSprop
- **Activation Functions:** ReLU, ELU, SELU, LeakyReLU
- **Epochs:** 30  
- **Batch Size:** 16
- **Callbacks:**
  - EarlyStopping
  - ReduceLROnPlateau
- **Data Split:** Train / Validation / Test (stratified when possible)

---

## 📊 Evaluation Metrics
Each model is evaluated using:
- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1-score (macro-averaged)

Results are aggregated into a **comparative benchmark table** for fair evaluation across architectures.

---

## 🚀 How to Run

### 1️⃣ Install Dependencies
```bash
pip install kagglehub tensorflow matplotlib scikit-learn opencv-python seaborn tqdm pandas
