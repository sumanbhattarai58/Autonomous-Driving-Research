import sys
sys.path.append(".")

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import create_dataloaders
from model_tuning import NvidiaModel, load_checkpoint
from train_tuning import config

with open("Nvidia_tuning/Results/best_checkpoint.txt", "r") as f:
    config.checkpoint_path = f.read().strip()

print(f"Loading best checkpoint: {config.checkpoint_path}")

# Collect all predictions and ground truth labels

def get_predictions(model, loader, device):
    model.eval()

    all_predictions = []
    all_actuals     = []

    with torch.no_grad():
        for images, angles in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            preds  = model(images)

            all_predictions.extend(preds.cpu().numpy())
            all_actuals.extend(angles.numpy())

    return np.array(all_predictions), np.array(all_actuals)


# Compute evaluation metrics

def compute_metrics(predictions, actuals):
    errors = predictions - actuals

    mse  = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(errors))

    #save metrics to a text file
    with open("Nvidia_tuning/Results/evaluation_metrics.txt", "w") as f:
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")


    return {"MSE": mse, "RMSE": rmse, "MAE": mae}


# Plot results

def plot_predictions(predictions, actuals, save_path="Nvidia_tuning/Results/evaluation_plot.png"):
    errors = predictions - actuals

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(actuals,     label="Actual",    alpha=0.7, linewidth=0.8)
    axes[0].plot(predictions, label="Predicted", alpha=0.7, linewidth=0.8)
    axes[0].set_title("Predicted vs Actual Steering Angle")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Steering Angle")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    axes[1].set_title("Prediction Error Distribution")
    axes[1].set_xlabel("Error (Predicted - Actual)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved → {save_path}")


def plot_losses():
    trials = {}
    current_trial = None

    with open("Nvidia_tuning/Results/losses.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("epoch"):
                continue
            if line.startswith("---"):
                current_trial = line  # use header as trial label
                trials[current_trial] = {"epochs": [], "train": [], "val": []}
                continue
            e, t, v = line.split(",")
            trials[current_trial]["epochs"].append(int(e))
            trials[current_trial]["train"].append(float(t))
            trials[current_trial]["val"].append(float(v))

    n = len(trials)
    fig, axes = plt.subplots(1, n, figsize=(12 * n, 3* n))
    if n == 1:
        axes = [axes]

    for ax, (trial_name, data) in zip(axes, trials.items()):
        ax.plot(data["epochs"], data["train"], label="Train Loss")
        ax.plot(data["epochs"], data["val"],   label="Val Loss")
        ax.set_title(trial_name.replace("---", "").strip(), fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("Nvidia_tuning/Results/training_validation_loss.png", dpi=150)
    plt.show()


# Main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    _, val_loader = create_dataloaders(config)
    print(f"Validation batches: {len(val_loader)}")

    print(f"Loading checkpoint: {config.checkpoint_path}")
    model = NvidiaModel().to(device)
    model, epoch = load_checkpoint(model, config.checkpoint_path, device)

    predictions, actuals = get_predictions(model, val_loader, device)

    metrics = compute_metrics(predictions, actuals)
    print("\n--- Evaluation Nvidia_tuning/Results ---")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")

    plot_predictions(predictions, actuals)
    plot_losses()


if __name__ == "__main__":
    main()