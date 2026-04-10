import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# 🔹 Import your model
from edcnn_model import EDCNN   # make sure file name is correct

# 🔹 Load model
@st.cache_resource
def load_model():
    model = EDCNN()
    model.load_state_dict(torch.load("edcnn.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 🔹 Title
st.title("🧠 Low Dose CT Image Denoising using EDCNN")

st.write("Upload a noisy CT scan image to get denoised output")

# 🔹 File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image_np = np.array(image)

    # Normalize
    image_norm = image_np / 255.0
    image_tensor = torch.tensor(image_norm).unsqueeze(0).unsqueeze(0).float()

    # 🔹 Inference
    with torch.no_grad():
        output = model(image_tensor)

    output = output.squeeze().numpy()
    output = np.clip(output, 0, 1)

    # 🔹 Convert to uint8 for OpenCV
    noisy_uint8 = (image_norm * 255).astype(np.uint8)
    denoised_uint8 = (output * 255).astype(np.uint8)

    # 🔹 Compute noise maps using Laplacian (high-frequency content)
    noisy_noise_map = cv2.Laplacian(noisy_uint8, cv2.CV_64F)
    denoised_noise_map = cv2.Laplacian(denoised_uint8, cv2.CV_64F)

    # 🔹 Take absolute values
    noisy_noise_map = np.abs(noisy_noise_map)
    denoised_noise_map = np.abs(denoised_noise_map)

    # 🔹 Normalize for visualization
    noisy_noise_map = (noisy_noise_map - noisy_noise_map.min()) / (noisy_noise_map.max() - noisy_noise_map.min() + 1e-8)
    denoised_noise_map = (denoised_noise_map - denoised_noise_map.min()) / (denoised_noise_map.max() - denoised_noise_map.min() + 1e-8)

    # 🔹 Display images
    # st.subheader("Results")

    # col1, col2 = st.columns(2)

    # with col1:
    #     st.image(image_np, caption="Noisy Image", use_container_width=True)


    # with col2:
    #     st.image(output, caption="Denoised Image", use_container_width=True)

    # 🔹 Display images
    # 🔹 Create heatmap-style visualization


    # st.subheader("Denoising Results")

    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     st.image(image_norm, caption="Noisy Image", use_container_width=True)

    # with col2:
    #     st.image(output, caption="Denoised Image", use_container_width=True)

    # with col3:
    #     st.image(noise_removed, caption="Removed Noise (Difference)", use_container_width=True, clamp=True)


   

    st.subheader("Noise Density Comparison")

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    # Row 1: Noisy Image + Heatmap
    axes[0, 0].imshow(image_norm, cmap='gray')
    axes[0, 0].set_title("Noisy Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_noise_map, cmap='hot')
    axes[0, 1].set_title("Noise Density (Before)")
    axes[0, 1].axis('off')

    # Row 2: Denoised Image + Heatmap
    axes[1, 0].imshow(output, cmap='gray')
    axes[1, 0].set_title("Denoised Image")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(denoised_noise_map, cmap='hot')
    axes[1, 1].set_title("Noise Density (After)")
    axes[1, 1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)