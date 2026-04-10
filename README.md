🧠 Medical Image Denoising using Deep Learning
📌 Project Title

Low-Dose CT Image Denoising using Deep Learning Models (EDCNN, RED-CNN, U-Net, DnCNN, GAN)

📖 Overview

Low-dose CT (LDCT) scans reduce patient radiation exposure but introduce heavy noise, affecting image quality and diagnosis.

This project focuses on reconstructing high-quality CT images from LDCT inputs using deep learning-based denoising models.

We implemented and compared multiple architectures to identify the best-performing model.

🎯 Objectives
Reduce noise in LDCT images
Improve image quality for diagnosis
Compare multiple deep learning models
Evaluate performance using PSNR and SSIM metrics
🧠 Models Implemented
1. EDCNN (Edge-Enhanced CNN)
Uses edge enhancement filters (Sobel-based)
Preserves structural details
Strong performance in medical images
2. RED-CNN
Residual Encoder-Decoder CNN
Uses skip connections
Good for medical image restoration
3. U-Net
Encoder-decoder architecture
Skip connections for feature preservation
Strong baseline model for segmentation & denoising
4. DnCNN
Deep CNN with residual learning
Focuses on noise estimation
Efficient for general image denoising
5. GAN (Generative Adversarial Network)
Generator + Discriminator framework
Learns realistic image reconstruction
Produces sharp outputs
📊 Dataset

We used:

📦 Low-Dose CT Reconstruction Dataset

Source: Kaggle
Type: Paired LDCT & NDCT images
Format: 512×512 CT slices
Split: Patient-wise train-test split
⚙️ Preprocessing
Image normalization (0–1 scaling)
Patch extraction (64×64 for training)
Data augmentation via random cropping
Grayscale conversion
🏗️ Workflow
LDCT Input Image
        ↓
Deep Learning Model (EDCNN / U-Net / etc.)
        ↓
Denoised CT Output
        ↓
Comparison with NDCT (Ground Truth)
📏 Evaluation Metrics

We used standard image quality metrics:

✔ PSNR (Peak Signal-to-Noise Ratio)
Measures reconstruction quality
Higher is better
✔ SSIM (Structural Similarity Index)
Measures structural similarity
Range: 0 to 1 (higher is better)
📈 Results Summary
Model	PSNR (dB)	SSIM
EDCNN	XX.XX	X.XXXX
RED-CNN	XX.XX	X.XXXX
U-Net	XX.XX	X.XXXX
DnCNN	XX.XX	X.XXXX
GAN	XX.XX	X.XXXX

👉 Best Model: EDCNN / (your best result)

💾 Saved Models

All trained models are saved in .pth format:

edcnn_model.pth
redcnn_model.pth
unet_model.pth
dncnn_model.pth
gan_model.pth
🚀 How to Run
1. Install dependencies
pip install torch torchvision numpy matplotlib scikit-image kagglehub pillow
2. Load dataset
import kagglehub
path = kagglehub.dataset_download("andrewmvd/ct-low-dose-reconstruction")
3. Load model
model = UNet()
model.load_state_dict(torch.load("unet_model.pth"))
model.eval()
4. Evaluate
evaluate(model, test_loader)
📊 Output Visualization
LDCT (Noisy input)
Denoised output
NDCT (Ground truth)
Error maps
🔬 Key Contributions
Implemented 5 deep learning architectures
Compared performance using medical metrics
Used patient-wise dataset splitting
Improved reconstruction quality of CT images
📌 Conclusion

Deep learning significantly improves low-dose CT image quality.
Among all models, residual and edge-enhanced architectures performed best in preserving structural details.

👨‍💻 Tech Stack
Python 🐍
PyTorch 🔥
NumPy
OpenCV
Matplotlib
KaggleHub
📂 Repository Structure
├── EDCNN/
├── REDCNN/
├── UNet/
├── DnCNN/
├── GAN/
├── models/
│   ├── edcnn.pth
│   ├── unet.pth
│   └── ...
├── dataset_loader.py
├── train.py
├── evaluate.py
└── README.md
⭐ Future Work
Improve GAN stability
Add attention mechanisms
Real-time CT denoising system
Web UI using Streamlit
