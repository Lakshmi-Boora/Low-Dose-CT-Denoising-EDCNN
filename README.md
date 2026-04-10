# 🧠 Medical Image Denoising using Deep Learning

---

## 📌 Project Title
Low-Dose CT Image Denoising using Deep Learning Models  
(EDCNN, RED-CNN, U-Net, DnCNN, GAN)

---

## 📖 Overview
Low-dose CT (LDCT) scans are widely used to reduce patient radiation exposure. However, they introduce heavy noise that reduces image quality and may affect diagnosis accuracy.

This project focuses on reconstructing high-quality Normal-Dose CT (NDCT) images from LDCT inputs using deep learning-based denoising techniques.

We implement multiple deep learning architectures and compare their performance using standard medical image quality metrics.

---

## 🎯 Objectives
- Reduce noise in low-dose CT images  
- Improve visual and diagnostic image quality  
- Implement multiple deep learning models  
- Compare models using PSNR and SSIM metrics  
- Identify the best-performing architecture  

---

## 🧠 Models Implemented

### 1. EDCNN (Edge-Enhanced Convolutional Neural Network)
EDCNN enhances edge information using Sobel filters before feature extraction.  
It helps preserve structural details in medical images.

- Edge-aware preprocessing  
- Strong structural preservation  
- Better sharpness in output images  

---

### 2. RED-CNN (Residual Encoder-Decoder CNN)
RED-CNN uses an encoder-decoder structure with residual learning.

- Skip connections between layers  
- Reduces information loss  
- Very effective for CT reconstruction  

---

### 3. U-Net
U-Net is a widely used architecture in medical imaging.

- Encoder-decoder structure  
- Skip connections for feature reuse  
- Strong baseline for segmentation and denoising  

---

### 4. DnCNN (Denoising CNN)
DnCNN learns residual noise instead of direct mapping.

- Residual learning strategy  
- Efficient and fast training  
- Strong general-purpose denoiser  

---

### 5. GAN (Generative Adversarial Network)
GAN uses a generator and discriminator to produce realistic outputs.

- Generator creates denoised images  
- Discriminator improves realism  
- Produces sharp outputs  

---

## 📊 Dataset

- **Name:** Low-Dose CT Reconstruction Dataset  
- **Source:** Kaggle  
- **Type:** Paired LDCT and NDCT images  
- **Image Size:** 512 × 512 CT slices  
- **Split:** Patient-wise train-test split  

Dataset link:
https://www.kaggle.com/datasets/andrewmvd/ct-low-dose-reconstruction

---

## ⚙️ Preprocessing Steps
- Converted images to grayscale  
- Normalized pixel values to [0, 1]  
- Extracted 64×64 patches for training  
- Applied random cropping augmentation  
- Organized patient-wise data split  

---

## 🏗️ Project Pipeline

LDCT Input Image  
↓  
Deep Learning Model (EDCNN / RED-CNN / U-Net / DnCNN / GAN)  
↓  
Denoised CT Output Image  
↓  
Comparison with NDCT (Ground Truth)  

---

## 📏 Evaluation Metrics

### ✔ PSNR (Peak Signal-to-Noise Ratio)
Measures the reconstruction quality of images.  
Higher PSNR indicates better image quality.

Formula:
PSNR = 10 × log10(MAX² / MSE)

---

### ✔ SSIM (Structural Similarity Index)
Measures structural similarity between two images.

- Range: 0 to 1  
- Higher value means better similarity  

---

## 📈 Results Comparison

| Model   | PSNR (dB) | SSIM  |
|----------|------------|--------|
| EDCNN   | XX.XX      | X.XXXX |
| RED-CNN | XX.XX      | X.XXXX |
| U-Net   | XX.XX      | X.XXXX |
| DnCNN   | XX.XX      | X.XXXX |
| GAN     | XX.XX      | X.XXXX |

👉 Best Performing Model: **EDCNN (replace with your actual result)**

---

## 💾 Saved Models (.pth files)

All trained models are saved in PyTorch format:

- edcnn_model.pth  
- redcnn_model.pth  
- unet_model.pth  
- dncnn_model.pth  
- gan_model.pth  

---

## 🚀 Installation & Setup

### Step 1: Install dependencies
```bash
pip install torch torchvision numpy matplotlib scikit-image kagglehub pillow opencv-python


### Step 2: Download dataset
import kagglehub

path = kagglehub.dataset_download("andrewmvd/ct-low-dose-reconstruction")
print("Dataset downloaded at:", path)


Step 3: Load model
import torch
from model import UNet

model = UNet()
model.load_state_dict(torch.load("unet_model.pth", map_location=torch.device('cpu')))
model.eval()


Step 4: Run evaluation
from evaluate import evaluate

results = evaluate(model, test_loader)
print(results)

📊 Output Visualization

The model generates the following outputs:

LDCT (Noisy input image)
Denoised CT image (Model output)
NDCT (Ground truth image)
Error difference map


🔬 Key Contributions
Implemented 5 deep learning architectures from scratch
Compared multiple models on medical dataset
Used patient-wise splitting for realistic evaluation
Improved reconstruction quality of CT scans
Analyzed performance using PSNR and SSIM

👨‍💻 Tech Stack
Python
PyTorch
NumPy
OpenCV
Matplotlib
Scikit-Image
KaggleHub

⭐ Future Improvements
Improve GAN stability with advanced loss functions
Add attention-based U-Net
Deploy as a web app using Streamlit
Optimize models for real-time inference
Extend to MRI and PET scan denoising
