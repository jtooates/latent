"""Utilities for visualizing and saving latent images."""

import torch
import numpy as np
from pathlib import Path
from typing import List, Optional
from PIL import Image


def normalize_latent_image(Z: torch.Tensor, method: str = 'minmax') -> torch.Tensor:
    """Normalize latent image tensor to [0, 1] range for visualization.

    Args:
        Z: Latent image tensor [B, C, H, W] or [C, H, W].
        method: Normalization method ('minmax' or 'standardize').

    Returns:
        Normalized tensor in [0, 1] range.
    """
    if method == 'minmax':
        # Normalize each image independently to [0, 1]
        Z_min = Z.flatten(start_dim=-2).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        Z_max = Z.flatten(start_dim=-2).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        Z_norm = (Z - Z_min) / (Z_max - Z_min + 1e-8)
    elif method == 'standardize':
        # Standardize to mean=0, std=1, then clip to [-3, 3] and scale to [0, 1]
        Z_mean = Z.flatten(start_dim=-2).mean(dim=-1, keepdim=True).unsqueeze(-1)
        Z_std = Z.flatten(start_dim=-2).std(dim=-1, keepdim=True).unsqueeze(-1)
        Z_norm = (Z - Z_mean) / (Z_std + 1e-8)
        Z_norm = torch.clamp(Z_norm, -3, 3)
        Z_norm = (Z_norm + 3) / 6  # Map [-3, 3] to [0, 1]
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return Z_norm


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized [C, H, W] tensor to PIL Image.

    Args:
        tensor: Tensor in [0, 1] range with shape [C, H, W].

    Returns:
        PIL Image in RGB or grayscale mode.
    """
    # Convert to numpy and scale to [0, 255]
    arr = (tensor.cpu().numpy() * 255).astype(np.uint8)

    # Handle different channel counts
    if arr.shape[0] == 1:
        # Grayscale: [1, H, W] -> [H, W]
        arr = arr[0]
        return Image.fromarray(arr, mode='L')
    elif arr.shape[0] == 3:
        # RGB: [3, H, W] -> [H, W, 3]
        arr = arr.transpose(1, 2, 0)
        return Image.fromarray(arr, mode='RGB')
    else:
        raise ValueError(f"Unsupported number of channels: {arr.shape[0]}")


def save_latent_image(
    Z: torch.Tensor,
    output_path: str,
    normalize: bool = True,
    norm_method: str = 'minmax',
    scale_size: int = None
):
    """Save a single latent image to disk.

    Args:
        Z: Latent image tensor [C, H, W].
        output_path: Path to save the image.
        normalize: Whether to normalize before saving.
        norm_method: Normalization method ('minmax' or 'standardize').
        scale_size: If provided, scale image to (scale_size, scale_size) using nearest neighbor.
    """
    if normalize:
        Z = normalize_latent_image(Z, method=norm_method)

    img = tensor_to_pil(Z)

    # Scale up if requested
    if scale_size is not None:
        img = img.resize((scale_size, scale_size), Image.NEAREST)

    img.save(output_path)


def save_latent_images(
    Z: torch.Tensor,
    sentences: List[str],
    output_dir: str,
    prefix: str = "",
    normalize: bool = True,
    norm_method: str = 'minmax',
    scale_size: int = None
):
    """Save multiple latent images with corresponding sentences.

    Args:
        Z: Batch of latent images [B, C, H, W].
        sentences: List of sentence strings (length B).
        output_dir: Directory to save images.
        prefix: Prefix for filenames (e.g., "epoch_10").
        normalize: Whether to normalize before saving.
        norm_method: Normalization method.
        scale_size: If provided, scale images to (scale_size, scale_size).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if normalize:
        Z = normalize_latent_image(Z, method=norm_method)

    for i, (z, sentence) in enumerate(zip(Z, sentences)):
        # Create filename from sentence (sanitize for filesystem)
        sentence_safe = sentence.replace(" ", "_").replace("/", "-")
        if len(sentence_safe) > 50:
            sentence_safe = sentence_safe[:50]

        filename = f"{prefix}_{i:03d}_{sentence_safe}.png" if prefix else f"{i:03d}_{sentence_safe}.png"
        filepath = output_dir / filename

        save_latent_image(z, str(filepath), normalize=False, scale_size=scale_size)  # Already normalized above


def create_image_grid(
    images: List[torch.Tensor],
    nrow: int = 4,
    padding: int = 2,
    pad_value: float = 1.0
) -> torch.Tensor:
    """Create a grid of images.

    Args:
        images: List of image tensors [C, H, W], all same size.
        nrow: Number of images per row.
        padding: Padding between images.
        pad_value: Value for padding (in [0, 1] range).

    Returns:
        Grid image tensor [C, H_grid, W_grid].
    """
    if not images:
        raise ValueError("images list is empty")

    # Stack images
    batch = torch.stack(images)  # [B, C, H, W]
    B, C, H, W = batch.shape

    # Compute grid dimensions
    ncol = (B + nrow - 1) // nrow  # Number of rows needed
    grid_h = ncol * H + (ncol + 1) * padding
    grid_w = nrow * W + (nrow + 1) * padding

    # Create grid tensor filled with pad_value
    grid = torch.ones((C, grid_h, grid_w)) * pad_value

    # Place images in grid
    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow

        y_start = row * H + (row + 1) * padding
        x_start = col * W + (col + 1) * padding

        grid[:, y_start:y_start + H, x_start:x_start + W] = img

    return grid


def visualize_latent_grid(
    Z: torch.Tensor,
    sentences: List[str],
    output_path: str,
    nrow: int = 4,
    normalize: bool = True,
    norm_method: str = 'minmax'
):
    """Create and save a grid visualization of multiple latent images.

    Args:
        Z: Batch of latent images [B, C, H, W].
        sentences: List of sentence strings (length B).
        output_path: Path to save the grid image.
        nrow: Number of images per row in the grid.
        normalize: Whether to normalize before saving.
        norm_method: Normalization method.
    """
    if normalize:
        Z = normalize_latent_image(Z, method=norm_method)

    # Convert batch to list of images
    images = [Z[i] for i in range(Z.size(0))]

    # Create grid
    grid = create_image_grid(images, nrow=nrow)

    # Save grid
    img = tensor_to_pil(grid)
    img.save(output_path)

    # Also save a text file with sentences
    text_path = Path(output_path).with_suffix('.txt')
    with open(text_path, 'w') as f:
        for i, sentence in enumerate(sentences):
            f.write(f"{i}: {sentence}\n")


def save_latent_statistics(
    Z: torch.Tensor,
    output_path: str,
    sentence: Optional[str] = None
):
    """Save statistics about a latent image for analysis.

    Args:
        Z: Latent image tensor [C, H, W] or [B, C, H, W].
        output_path: Path to save statistics text file.
        sentence: Optional sentence that generated this latent.
    """
    if Z.dim() == 4:
        # Batch: compute stats across spatial dimensions only
        Z_flat = Z.flatten(start_dim=-2)  # [B, C, H*W]
        mean = Z_flat.mean(dim=-1)  # [B, C]
        std = Z_flat.std(dim=-1)  # [B, C]
        min_val = Z_flat.min(dim=-1)[0]  # [B, C]
        max_val = Z_flat.max(dim=-1)[0]  # [B, C]
    else:
        # Single image
        Z_flat = Z.flatten(start_dim=-2)  # [C, H*W]
        mean = Z_flat.mean(dim=-1)  # [C]
        std = Z_flat.std(dim=-1)  # [C]
        min_val = Z_flat.min(dim=-1)[0]  # [C]
        max_val = Z_flat.max(dim=-1)[0]  # [C]

    with open(output_path, 'w') as f:
        if sentence:
            f.write(f"Sentence: {sentence}\n\n")

        f.write("Per-channel statistics:\n")
        for c in range(Z.size(-3)):  # Channel dimension
            if Z.dim() == 4:
                f.write(f"  Channel {c}:\n")
                for b in range(Z.size(0)):
                    f.write(f"    Batch {b}: mean={mean[b, c]:.4f}, std={std[b, c]:.4f}, "
                           f"min={min_val[b, c]:.4f}, max={max_val[b, c]:.4f}\n")
            else:
                f.write(f"  Channel {c}: mean={mean[c]:.4f}, std={std[c]:.4f}, "
                       f"min={min_val[c]:.4f}, max={max_val[c]:.4f}\n")
