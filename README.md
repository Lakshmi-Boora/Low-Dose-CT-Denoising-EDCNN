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

| Model          | PSNR (dB)   | SSIM   |
|----------------|-------------|--------|
| EDCNN          | 42.55       | 0.9644 |
| RED-CNN        | 41.92       | 0.9623 |
| U-Net          |40.90        | 0.9551 |
| DnCNN          | 28.19       | 0.6349 |
| WGAN           | 40.82       |0.9465  |
|Attention U-Net |36.63        |0.9488  |
  

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

<!-- 🚀 INSTALLATION & SETUP -->
<h2>🚀 Installation & Setup</h2>

<h3>Step 1: Install dependencies</h3>
<pre>
pip install torch torchvision numpy matplotlib scikit-image kagglehub pillow opencv-python
</pre>

<h3>Step 2: Download dataset</h3>
<pre>
import kagglehub

path = kagglehub.dataset_download("andrewmvd/ct-low-dose-reconstruction")
print("Dataset downloaded at:", path)
</pre>

<h3>Step 3: Load model</h3>
<pre>
import torch
from model import UNet

model = UNet()
model.load_state_dict(torch.load("unet_model.pth", map_location=torch.device('cpu')))
model.eval()
</pre>

<h3>Step 4: Run evaluation</h3>
<pre>
from evaluate import evaluate

results = evaluate(model, test_loader)
print(results)
</pre>

<hr>

<!-- 📊 OUTPUT VISUALIZATION -->
<h2>📊 Output Visualization</h2>
<ul>
  <li>LDCT (Noisy input image)</li>
  <li>Denoised CT image (Model output)</li>
  <li>NDCT (Ground truth image)</li>
  <li>Error difference map</li>
</ul>

<hr>

<!-- 🔬 KEY CONTRIBUTIONS -->
<h2>🔬 Key Contributions</h2>
<ul>
  <li>Implemented 5 deep learning architectures from scratch</li>
  <li>Compared multiple models on medical dataset</li>
  <li>Used patient-wise splitting for realistic evaluation</li>
  <li>Improved reconstruction quality of CT scans</li>
  <li>Analyzed performance using PSNR and SSIM</li>
</ul>

<hr>

<!-- 👨‍💻 TECH STACK -->
<h2>👨‍💻 Tech Stack</h2>

<div style="display:flex; gap:10px; flex-wrap:wrap;">
  <span>🐍 Python</span>
  <span>🔥 PyTorch</span>
  <span>📊 NumPy</span>
  <span>🖼 OpenCV</span>
  <span>📈 Matplotlib</span>
  <span>🧪 Scikit-Image</span>
  <span>📦 KaggleHub</span>
</div>

<hr>

<!-- ⭐ FUTURE IMPROVEMENTS -->
<h2>⭐ Future Improvements</h2>
<ul>
  <liTransformer-Based Architectures
/li>
  <li>ADiffusion Model-Based Denoising</li>
  <li>Advanced Loss Functions
</li>
  <li>Clinical Validation</li>
  <li>Extend to MRI and PET scan denoising</li>
</ul>

👨‍💻 Tech Stack
Python
PyTorch
NumPy
OpenCV
Matplotlib
Scikit-Image
KaggleHub

⭐ Future Improvements
Transformer-Based Architectures
Advanced Loss Functions
Diffusion Model-Based Denoising
Clinical Validation
