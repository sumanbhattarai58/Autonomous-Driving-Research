# import necessary libraries
import random
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Define the expected columns in the CSV file
CSV_COLUMNS = ["center", "left", "right", "steering", "throttle", "brake", "speed"]

# Load Image 
def load_image(image_path):
    """Load an image from disk and return it as RGB numpy array."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def strip_image_path(data_dir, raw_path):
    """
    Fix image paths from the CSV.
    The Udacity CSV stores full paths from the recording machine,
    so we just take the filename and look for it inside data_dir/IMG/.
    """
    filename = Path(raw_path.strip()).name
    path = Path(data_dir) / "IMG" / filename
    if not path.exists():
        # fallback: try treating raw_path as relative to data_dir
        path = Path(data_dir) / raw_path.strip()
    return path


# Image preprocessing

def crop_image(image, top_crop, bottom_crop):
    """Remove sky from top and car hood from bottom."""
    h = image.shape[0]
    bottom = h - bottom_crop if bottom_crop > 0 else h
    return image[top_crop:bottom, :, :]


def convert_rgb_to_yuv(image):
    """Convert RGB image to YUV (as used in the NVIDIA paper).
    Contains brightness and color information"""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)


def normalize_image(image):
    """Scale pixel values from [0, 255] to [-1, 1]."""
    return (image.astype(np.float32) / 127.5) - 1.0


def to_tensor(image):
    """Convert HxWxC numpy array to CxHxW torch tensor."""
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


# Augmentation pipelines (Albumentations)

def train_augmentations(img_height, img_width):
    """Augmentations applied during training only."""
    return A.Compose([
        # Slight random crop before resize (simulates camera jitter)
        A.RandomResizedCrop(
            size=(img_height, img_width),
            scale=(0.80, 0.95),
            ratio=(2.5, 3.5),
            p=0.5
        ),
        # Always resize to NVIDIA input size
        A.Resize(height=img_height, width=img_width),
        # Flip image horizontally (steering angle must be negated when this happens)
        A.HorizontalFlip(p=0.6),
        # Random brightness and contrast changes
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.2),
        # Simulate shadows (e.g. from trees, bridges)
        A.RandomShadow(shadow_roi=(0, 0.4, 1, 1), num_shadows_limit=(1, 2), p=0.2),
        # Slight blur to simulate motion or camera noise
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ])


def val_augmentations(img_height, img_width):
    """Only resize during validation — no random augmentations."""
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
    ])


def check_if_flipped(augmentation_result):
    """
    Check if a horizontal flip was applied during augmentation.
    We need this to correctly negate the steering angle.
    Albumentations stores applied transform info in result['replay'].
    """
    replay = augmentation_result.get("replay", {})
    for transform in replay.get("transforms", []):
        name = transform.get("__class_fullname__", "")
        if "HorizontalFlip" in name and transform.get("applied", False):
            return True
    return False


# Camera selection

def select_camera(row, data_dir, is_training, steering_correction):
    """
    Pick one of three cameras (center, left, right).
    - Training: pick randomly, adjust steering angle for left/right
    - Validation: always use center camera only
    """
    base_angle = float(row["steering"])

    if not is_training:
        image = load_image(strip_image_path(data_dir, row["center"]))
        return image, base_angle
    
    #Training: randomly pick one of the three cameras
    camera = random.choice(["center", "left", "right"])
    image = load_image(strip_image_path(data_dir, row[camera]))

    if camera == "left":
        angle = base_angle + steering_correction   # left camera: steer more right
    elif camera == "right":
        angle = base_angle - steering_correction   # right camera: steer more left
    else:
        angle = base_angle

    return image, angle


# Data balancing

def reduce_zero_steering(df, threshold=0.05, keep=0.30):
    """
    The car drives straight most of the time, which creates too many
    near-zero steering samples. This removes most of them to help balance
    the dataset and prevent the model from just learning to go straight.
    """
    is_zero = df["steering"].abs() < threshold
    df_zero = df[is_zero].sample(frac=keep, random_state=42)
    df_nonzero = df[~is_zero]
    return pd.concat([df_zero, df_nonzero]).reset_index(drop=True)


def histogram_balance(df, n_bins=25):
    """
    Balance steering angle distribution by limiting how many samples
    each steering angle bin can have. Bins with too many samples are
    randomly downsampled to the average count across all bins.
    """
    angles = df["steering"].values
    hist, bin_edges = np.histogram(angles, bins=n_bins)
    max_per_bin = int(np.mean(hist))

    keep = []
    rng = np.random.RandomState(42)

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        # Include right edge for last bin
        if i == n_bins - 1:
            idx = np.where((angles >= low) & (angles <= high))[0]
        else:
            idx = np.where((angles >= low) & (angles < high))[0]

        if len(idx) > max_per_bin:
            idx = rng.choice(idx, size=max_per_bin, replace=False)

        keep.extend(idx.tolist())

    return df.iloc[keep].reset_index(drop=True)


def compute_sample_weights(df, n_bins=25):
    """
    Compute a weight for each sample so that rare steering angles
    are sampled more often during training (used with WeightedRandomSampler).
    """
    angles = df["steering"].values
    hist, bin_edges = np.histogram(angles, bins=n_bins)
    hist = np.maximum(hist, 1)  # avoid division by zero

    bin_idx = np.clip(np.digitize(angles, bin_edges[:-1]) - 1, 0, n_bins - 1)
    weights = 1.0 / hist[bin_idx]
    weights = weights / weights.sum()  # normalize so they sum to 1

    return torch.tensor(weights, dtype=torch.float32)


# Dataset class

class BehavioralCloningDataset(Dataset):
    """
    PyTorch Dataset for steering angle regression.

    Each sample returns:
        image  : torch.Tensor of shape (3, 66, 200) — normalized YUV image
        angle  : torch.Tensor scalar — steering angle in [-1, 1]

    During training:
        - Randomly picks center, left, or right camera
        - Applies full augmentation pipeline
        - Negates steering angle if image was flipped horizontally

    During validation:
        - Always uses center camera
        - Only resizes the image, no augmentations
    """

    def __init__(self, df, data_dir, is_training,
                 steering_correction=0.2,
                 top_crop=40, bottom_crop=20,
                 img_height=66, img_width=200):

        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.is_training = is_training
        self.steering_correction = steering_correction
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop

        # Load the correct augmentation pipeline
        if is_training:
            self.augment = train_augmentations(img_height, img_width)
        else:
            self.augment = val_augmentations(img_height, img_width)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Step 1: Pick camera and get steering angle
        image, angle = select_camera(
            row, self.data_dir, self.is_training, self.steering_correction
        )

        # Step 2: Crop sky and hood
        image = crop_image(image, self.top_crop, self.bottom_crop)

        # Step 3: Apply augmentations
        result = self.augment(image=image)
        image = result["image"]

        # Step 4: If image was flipped, negate the steering angle
        if self.is_training and check_if_flipped(result):
            angle = -angle

        # Step 5: Convert to YUV and normalize to [-1, 1]
        image = convert_rgb_to_yuv(image)
        image = normalize_image(image)

        # Step 6: Convert to tensor (3, 66, 200)
        image_tensor = to_tensor(image)
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        return image_tensor, angle_tensor


# DataLoader factory

def create_dataloaders(config):
    """
    Read the CSV, split into train/val, optionally balance the training set,
    and return two DataLoader objects ready for training.

    config fields:
        data_dir, batch_size, num_workers,
        steering_correction, top_crop, bottom_crop,
        img_height, img_width, val_split,
        use_balancing, use_weighted_sampler
    """

    # --- Load CSV ---
    csv_path = Path(config.data_dir) / "driving_log.csv"
    df = pd.read_csv(csv_path, names=CSV_COLUMNS, header=None, skipinitialspace=True)
    df = df.dropna(subset=["center", "left", "right", "steering"]).reset_index(drop=True)

    # --- Shuffle ---
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # --- Split into train and validation ---
    val_size = int(len(df) * config.val_split)
    df_val   = df.iloc[:val_size].reset_index(drop=True)
    df_train = df.iloc[val_size:].reset_index(drop=True)

    # --- Balance training data
    if config.use_balancing:
        df_train = reduce_zero_steering(df_train)
        df_train = histogram_balance(df_train)
        df_train = df_train.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # --- Create Dataset objects ---
    shared_args = dict(
        data_dir            = config.data_dir,
        steering_correction = config.steering_correction,
        top_crop            = config.top_crop,
        bottom_crop         = config.bottom_crop,
        img_height          = config.img_height,
        img_width           = config.img_width,
    )

    train_dataset = BehavioralCloningDataset(df_train, is_training=True,  **shared_args)
    val_dataset   = BehavioralCloningDataset(df_val,   is_training=False, **shared_args)

    # --- Optional: WeightedRandomSampler for further balancing ---
    sampler = None
    if config.use_weighted_sampler:
        weights = compute_sample_weights(df_train)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size   = config.batch_size,
        shuffle      = (sampler is None),   # shuffle and sampler can't both be set
        sampler      = sampler,
        num_workers  = config.num_workers,
        pin_memory   = torch.cuda.is_available(),
        drop_last    = True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = config.batch_size,
        shuffle     = False,
        num_workers = config.num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = False,
    )

    return train_loader, val_loader 
