
from importlib.resources import path
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from types import SimpleNamespace
from tqdm import tqdm

from dataset import create_dataloaders
from model import NvidiaModel, count_parameters


# Config 

config = SimpleNamespace(
    data_dir        = r"datasets\data\self_driving_car_dataset_jungle", 
    save_dir        = "checkpoints",                  

    # --- Training ---
    epochs          = 20,
    batch_size      = 32,
    learning_rate   = 1e-4,
    num_workers     = 2,


    # --- Dataset ---
    steering_correction  = 0.2,
    top_crop             = 40,
    bottom_crop          = 20,
    img_height           = 66,
    img_width            = 200,
    val_split            = 0.2,
    use_balancing        = True,
    use_weighted_sampler = False,
)


# One epoch of training

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for images, angles in tqdm(loader, desc="  Training", leave=False):
        images = images.to(device)
        angles = angles.to(device)

        # Forward pass
        predictions = model(images)
        loss = loss_fn(predictions, angles)

        # Backward pass
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

# clear previous checkpoints .pt files

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


# Main training loop

def main():
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # --- Model ---
    model = NvidiaModel().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # --- Loss, optimizer, scheduler ---
    loss_fn   = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=True)

    clear_previous_checkpoints(config.save_dir)

    # --- Training loop ---
    best_val_loss = float("inf")
    print(f"\nStarting training for {config.epochs} epochs...\n")

    open("Results/losses.txt", "w").close()  # clear previous losses

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss   = validate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        with open("Results/losses.txt", "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = save_checkpoint(model, optimizer, epoch, val_loss, config.save_dir)
            print(f"New best model saved → {path}")

            open("Results/best_checkpoint.txt", "w").close()
            
            # save best checkpoint name to a text file
            with open("Results/best_checkpoint.txt", "a") as f:
                f.write(path + "\n")

        print()

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")



if __name__ == "__main__":
    main()