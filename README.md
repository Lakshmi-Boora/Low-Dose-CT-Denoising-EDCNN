# 🧠 Medical Image Denoising using Deep Learning

## 📌 Project Title
Low-Dose CT Image Denoising using Deep Learning Models  
(EDCNN, RED-CNN, U-Net, DnCNN, GAN)

---

## 📖 Overview
Low-dose CT (LDCT) scans reduce patient radiation exposure but introduce heavy noise, affecting image quality and diagnosis.

This project focuses on reconstructing high-quality CT images from LDCT inputs using deep learning-based denoising models.

We implemented and compared multiple architectures to identify the best-performing model.

---

## 🎯 Objectives
- Reduce noise in LDCT images  
- Improve image quality for diagnosis  
- Compare multiple deep learning models  
- Evaluate performance using PSNR and SSIM metrics  

---

## 🧠 Models Implemented

### 1. EDCNN (Edge-Enhanced CNN)
- Uses Sobel-based edge enhancement
- Preserves structural details
- Strong performance in medical imaging

### 2. RED-CNN
- Residual Encoder-Decoder CNN
- Uses skip connections
- Effective for medical image restoration

### 3. U-Net
- Encoder-decoder architecture
- Skip connections for feature preservation
- Strong baseline for denoising tasks

### 4. DnCNN
- Deep CNN with residual learning
- Focuses on noise estimation
- Efficient and widely used

### 5. GAN
- Generator + Discriminator framework
- Produces realistic reconstructed images
- Sharp output quality

---

## 📊 Dataset

- **Dataset:** Low-Dose CT Reconstruction Dataset  
- **Source:** Kaggle  
- **Type:** Paired LDCT & NDCT images  
- **Format:** 512×512 CT slices  
- **Split:** Patient-wise train-test split  

---

## ⚙️ Preprocessing
- Image normalization (0–1 scaling)  
- Patch extraction (64×64)  
- Random cropping augmentation  
- Grayscale conversion  

---

## 🏗️ Workflow

---


## 📏 Evaluation Metrics

### ✔ PSNR (Peak Signal-to-Noise Ratio)
- Measures reconstruction quality  
- Higher value = better performance  

### ✔ SSIM (Structural Similarity Index)
- Measures structural similarity  
- Range: 0 to 1 (higher is better)  

---

## 📈 Results Summary

| Model   | PSNR (dB) | SSIM  |
|----------|------------|--------|
| EDCNN   | XX.XX      | X.XXXX |
| RED-CNN | XX.XX      | X.XXXX |
| U-Net   | XX.XX      | X.XXXX |
| DnCNN   | XX.XX      | X.XXXX |
| GAN     | XX.XX      | X.XXXX |

👉 **Best Model:** EDCNN / (your best result)

---

## 💾 Saved Models

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install torch torchvision numpy matplotlib scikit-image kagglehub pillow
