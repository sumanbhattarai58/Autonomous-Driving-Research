# inference.py
# Load a trained model and predict steering angle on a single image.
# Run with: python inference.py

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from model import NvidiaModel, load_checkpoint
from train import config
from dataset import load_image, strip_image_path, crop_image, convert_rgb_to_yuv, normalize_image


# ============================================================
# Config — only change these two lines
# ============================================================

with open("Results/best_checkpoint.txt", "r") as f:
    config.checkpoint_path = f.read().strip()

def load_samples_from_csv(data_direc, n=10):
    """Load n random samples from CSV with image path and actual steering angle."""
    csv_path = Path(data_direc) / "driving_log.csv"
    df = pd.read_csv(csv_path, names=["center","left","right","steering","throttle","brake","speed"], skipinitialspace=True)
    return df.sample(n=n, random_state=42).reset_index(drop=True)
RESULT_PATH = "Results/inference_result.png"

data_direc = Path(r"datasets\data\self_driving_car_dataset_make")
image_path = data_direc / "IMG"  # 👈 path to your image

# ============================================================
# Preprocess image — same steps as dataset.py
# ============================================================

def preprocess(image_path):
    rgb = load_image(Path(image_path))

    # crop sky and hood
    cropped = crop_image(rgb, config.top_crop, config.bottom_crop)

    # resize
    resized = cv2.resize(cropped, (config.img_width, config.img_height))

    # RGB to YUV and normalize
    yuv        = convert_rgb_to_yuv(resized)
    normalized = normalize_image(yuv)

    # to tensor
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return resized, tensor


# ============================================================
# Save result image with steering angle printed on it
# ============================================================

def save_result(original_img, preprocessed_img, angle, result_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(preprocessed_img)
    axes[1].set_title(f"Preprocessed Input\nPredicted Steering Angle: {angle:.4f}")
    axes[1].axis("off")

    plt.suptitle(f"Inference Result — Steering Angle: {angle:.4f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(result_path, dpi=150)
    plt.show()
    print(f"Result saved → {result_path}")


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NvidiaModel().to(device)
    model, epoch = load_checkpoint(model, config.checkpoint_path, device)
    model.eval()

    # load 10 random samples from CSV
    df = load_samples_from_csv(data_direc, n=10)
    results = []

    for _, row in df.iterrows():
        image_path   = strip_image_path(data_direc, row["center"])
        actual_angle = float(row["steering"])

        original_rgb         = load_image(image_path)
        preprocessed, tensor = preprocess(str(image_path))
        tensor               = tensor.to(device)

        with torch.no_grad():
            predicted_angle = model(tensor).item()

        error = abs(predicted_angle - actual_angle)
        print(f"Actual: {actual_angle:.4f} | Predicted: {predicted_angle:.4f} | Error: {error:.4f}")
        results.append((image_path.name, original_rgb, preprocessed, actual_angle, predicted_angle, error))

    # save to txt
    with open("Results/inference_result.txt", "w") as f:
        f.write("Image, Actual, Predicted, Error\n")
        for name, _, _, actual, predicted, error in results:
            f.write(f"{name} | Actual: {actual:.4f} | Predicted: {predicted:.4f} | Error: {error:.4f}\n")

    # plot
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))

    for i, (name, original_rgb, preprocessed, actual, predicted, error) in enumerate(results):
        axes[i][0].imshow(original_rgb)
        axes[i][0].set_title(f"{name}")
        axes[i][0].axis("off")

        axes[i][1].imshow(preprocessed)
        axes[i][1].set_title(f"Actual: {actual:.4f} | Predicted: {predicted:.4f} | Error: {error:.4f}")
        axes[i][1].axis("off")

    plt.suptitle("Inference — Predicted vs Actual Steering Angle", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULT_PATH, dpi=150)
    plt.show()
    print(f"Result saved → {RESULT_PATH}")


if __name__ == "__main__":
    main()