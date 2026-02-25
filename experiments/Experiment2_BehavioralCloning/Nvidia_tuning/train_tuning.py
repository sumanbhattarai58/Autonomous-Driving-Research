import sys
sys.path.append(".")

import os
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from types import SimpleNamespace
from tqdm import tqdm

from dataset import create_dataloaders
from model_tuning import NvidiaModel, count_parameters


# Config

config = SimpleNamespace(
    data_dir        = r"datasets\data\self_driving_car_dataset_jungle",
    save_dir        = "Nvidia_tuning/checkpoints",

    # --- Training ---
    epochs          = 20,
    batch_size      = 16,
    learning_rate   = 3e-4,
    weight_decay    = 1e-5,
    num_workers     = 2,

    # --- Dataset ---
    steering_correction  = 0.2,
    top_crop             = 40,
    bottom_crop          = 20,
    img_height           = 66,
    img_width            = 200,
    val_split            = 0.2,
    use_balancing        = True,
    use_weighted_sampler = True,

    checkpoint_path = "",

    # --- RandomizedSearch ---
    n_search_trials = 5,       # how many random combinations to try

    # --- Early Stopping ---
    early_stopping_patience = 7,  # stop if val loss doesn't improve for this many epochs
)


# Hyperparameter search space

SEARCH_SPACE = {
    "learning_rate" : [1e-4, 3e-4, 1e-3, 3e-3],
    "weight_decay"  : [1e-6, 1e-5, 1e-4],
    "batch_size"    : [16, 32],
    "dropout"       : [0.3, 0.5],
}


# One epoch of training

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for images, angles in tqdm(loader, desc="  Training", leave=False):
        images = images.to(device)
        angles = angles.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, angles)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# One epoch of validation

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, angles in tqdm(loader, desc="  Validating", leave=False):
            images = images.to(device)
            angles = angles.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, angles)
            total_loss += loss.item()

    return total_loss / len(loader)


# Clear previous checkpoints

def clear_previous_checkpoints(save_dir):
    for file in os.listdir(save_dir):
        if file.endswith(".pt"):
            os.remove(os.path.join(save_dir, file))


# Save checkpoint

def save_checkpoint(model, optimizer, epoch, val_loss, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"model_epoch{epoch:02d}_loss{val_loss:.4f}.pt")
    torch.save({
        "epoch":      epoch,
        "val_loss":   val_loss,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, path)
    return path


# Sample random hyperparameters from search space

def sample_hyperparams():
    return {key: random.choice(values) for key, values in SEARCH_SPACE.items()}


# Train one full trial with given hyperparameters (with early stopping)

def run_trial(trial_config, device, trial_num, hyperparams):
    print(f"\n--- Trial {trial_num} ---")
    print(f"  lr={trial_config.learning_rate} | weight_decay={trial_config.weight_decay} | batch_size={trial_config.batch_size} | dropout={trial_config.dropout}")

    train_loader, val_loader = create_dataloaders(trial_config)

    model     = NvidiaModel(dropout=trial_config.dropout).to(device)
    loss_fn   = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=trial_config.learning_rate, weight_decay=trial_config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=False)

    best_val_loss     = float("inf")
    best_model_path   = None
    no_improve_count  = 0

    # write trial header with hyperparams
    with open("Nvidia_tuning/Results/losses.txt", "a") as f:
        f.write(f"\n--- Trial {trial_num} | lr={trial_config.learning_rate} | weight_decay={trial_config.weight_decay} | batch_size={trial_config.batch_size} | dropout={trial_config.dropout} ---\n")
        f.write("epoch,train_loss,val_loss\n")

    for epoch in range(1, trial_config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss   = validate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        print(f"  Epoch {epoch}/{trial_config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        with open("Nvidia_tuning/Results/losses.txt", "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_path  = save_checkpoint(model, optimizer, epoch, val_loss, trial_config.save_dir)
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"  No improvement for {no_improve_count}/{trial_config.early_stopping_patience} epochs")

            if no_improve_count >= trial_config.early_stopping_patience:
                print(f"  Early stopping triggered at epoch {epoch}")
                break

    return best_val_loss, best_model_path


# Main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs("Nvidia_tuning/Results", exist_ok=True)
    clear_previous_checkpoints(config.save_dir)

    # --- Randomized Search ---
    print(f"\nStarting Randomized Search with {config.n_search_trials} trials...\n")

    best_overall_loss = float("inf")
    best_overall_path = None
    best_hyperparams  = None

    # clear losses file once before all trials start
    open("Nvidia_tuning/Results/losses.txt", "w").close()

    for trial_num in range(1, config.n_search_trials + 1):

        # sample random hyperparameters
        hyperparams = sample_hyperparams()

        # build trial config from base config + sampled hyperparams
        trial_config = SimpleNamespace(**vars(config))
        trial_config.learning_rate = hyperparams["learning_rate"]
        trial_config.weight_decay  = hyperparams["weight_decay"]
        trial_config.batch_size    = hyperparams["batch_size"]
        trial_config.dropout       = hyperparams["dropout"]

        trial_loss, trial_path = run_trial(trial_config, device, trial_num, hyperparams)

        print(f"  Trial {trial_num} best val loss: {trial_loss:.4f}")

        # track best overall trial
        if trial_loss < best_overall_loss:
            best_overall_loss = trial_loss
            best_overall_path = trial_path
            best_hyperparams  = hyperparams

    # --- Save best checkpoint info ---
    with open("Nvidia_tuning/Results/best_checkpoint.txt", "w") as f:
        f.write(best_overall_path)

    #save best  hyperparameters
    with open("Nvidia_tuning/Results/best_hyperparams.txt", "w") as f:
        for key, value in best_hyperparams.items():
            f.write(f"{key}: {value}\n")


    print(f"\nRandomized Search complete.")
    print(f"Best val loss : {best_overall_loss:.4f}")
    print(f"Best checkpoint: {best_overall_path}")
    print(f"Best hyperparams: {best_hyperparams}")



if __name__ == "__main__":
    main()