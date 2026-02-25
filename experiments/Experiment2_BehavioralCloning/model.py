import torch
import torch.nn as nn

# NVIDIA Model

class NvidiaModel(nn.Module):
    """
    5 convolutional layers learn to detect road features (lanes, edges, curves).
    4 fully connected layers learn to map those features to a steering angle.
    """

    def __init__(self, dropout=0.0):
        super().__init__()

        # --- Convolutional layers ---
        # Each layer learns increasingly abstract features:
        #   early layers → edges and textures
        #   later layers → curves, lane markings
        self.conv_layers = nn.Sequential(
            # Input: (batch, 3, 66, 200)
            nn.Conv2d(3,  24, kernel_size=5, stride=2),  # -> (batch, 24, 31, 98)
            nn.ELU(),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),  # -> (batch, 36, 14, 47)
            nn.ELU(),

            nn.Conv2d(36, 48, kernel_size=5, stride=2),  # -> (batch, 48,  5, 22)
            nn.ELU(),

            nn.Conv2d(48, 64, kernel_size=3),             # -> (batch, 64,  3, 20)
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3),             # -> (batch, 64,  1, 18)
            nn.ELU(),
        )

        # --- Fully connected layers ---
        # Flatten conv output (64 * 1 * 18 = 1152) then shrink down to 1 value
        self.fc_layers = nn.Sequential(
            nn.Flatten(),              # (batch, 1152)

            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Dropout(p=dropout),    # randomly zero neurons to prevent overfitting

            nn.Linear(100, 50),
            nn.ELU(),
            nn.Dropout(p=dropout),

            nn.Linear(50, 10),
            nn.ELU(),

            nn.Linear(10, 1),         # final output: one steering angle per image
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.squeeze(1)           # shape: (batch,) not (batch, 1)


# Helper: count model parameters

def count_parameters(model):
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Helper: load a saved checkpoint back into the model

def load_checkpoint(model, checkpoint_path, device):
    """
    Load a previously saved model checkpoint.
    Returns the model with loaded weights and the epoch it was saved at.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    epoch    = checkpoint.get("epoch", 0)
    val_loss = checkpoint.get("val_loss", None)
    print(f"Loaded checkpoint from epoch {epoch} | val loss: {val_loss:.4f}")
    return model, epoch


# Quick test — run this file directly to verify shapes

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = NvidiaModel().to(device)
    print(f"Total parameters: {count_parameters(model):,}")

    # Simulate a batch of 4 images
    dummy_input = torch.randn(4, 3, 66, 200).to(device)
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")   # (4, 3, 66, 200)
    print(f"Output shape: {output.shape}")         # (4,)
    print("Model is working correctly")