#!/usr/bin/env python3
"""Visualize saliency maps for latent bottleneck model.

This script generates pixel-level attribution maps showing which parts of the
latent image are most important for each property prediction (color1, size1, etc.).

Uses Gradient × Input attribution method with per-channel and combined visualizations.
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from typing import Dict, Tuple

from vocab import Vocabulary, Tokenizer, build_vocab_from_data
from model import (
    SimpleTextEncoder, ImagePropertyHead, FullModelWithBottleneck,
    CanvasPainterEncoder, FullModelWithCanvasPainter
)
from dataset import PropertyEncoder
from image_utils import normalize_latent_image


def load_model(checkpoint_path: str, data_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        data_path: Path to training data JSON (contains vocab and property encoder metadata).
        device: Device to load model on.

    Returns:
        Tuple of (model, tokenizer, property_encoder, config).
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Build vocabulary from data file
    vocab = build_vocab_from_data(data_path)
    tokenizer = Tokenizer(vocab, max_len=12)

    # Load property encoder from data file
    import json
    with open(data_path, 'r') as f:
        data = json.load(f)
    metadata = data['metadata']
    property_encoder = PropertyEncoder(
        colors=metadata['colors'],
        sizes=metadata['sizes'],
        shapes=metadata['shapes'],
        rels=metadata['rels']
    )
    print(f"  Loaded vocab and property encoder from data file: {data_path}")

    print(f"  Property encoder: n_colors={property_encoder.n_colors}, n_sizes={property_encoder.n_sizes}, n_shapes={property_encoder.n_shapes}, n_rels={property_encoder.n_rels}")

    # Create model - get hyperparameters from config or top-level checkpoint
    latent_size = config.get('latent_size') or checkpoint.get('latent_size', 32)
    latent_channels = config.get('latent_channels') or checkpoint.get('latent_channels', 3)
    use_maxpool = config.get('use_maxpool') if config.get('use_maxpool') is not None else checkpoint.get('use_maxpool', True)
    pool_size = config.get('pool_size') or checkpoint.get('pool_size', 2)

    # Check if this is a canvas painter model
    use_canvas_painter = 'canvas_painter' in str(checkpoint.get('config', {})) or \
                         any('canvas_painter' in key for key in checkpoint['model_state_dict'].keys())

    # Create property head (same for both architectures)
    head = ImagePropertyHead(
        img_channels=latent_channels,
        n_colors=property_encoder.n_colors,
        n_sizes=property_encoder.n_sizes,
        n_shapes=property_encoder.n_shapes,
        n_rels=property_encoder.n_rels,
        conv_channels=(32, 64),
        hidden_dim=128,
        use_maxpool=use_maxpool,
        pool_size=pool_size
    )

    if use_canvas_painter:
        print("  Detected canvas painter model")

        # Create encoder (no latent_dim for canvas painter)
        encoder = SimpleTextEncoder(
            num_tokens=len(vocab),
            max_len=12,
            d_model=128,
            nhead=4,
            ff_dim=256,
            num_layers=2,
            dropout=0.1
        )

        # Create canvas painter
        painter_d_state = config.get('painter_d_state', 256)
        painter_patch_size = config.get('painter_patch_size', 5)
        painter_num_steps = config.get('painter_num_steps', 0)

        canvas_painter = CanvasPainterEncoder(
            d_model=128,
            d_state=painter_d_state,
            H=latent_size,
            W=latent_size,
            C=latent_channels,
            K=painter_patch_size,
            num_steps=painter_num_steps
        )

        # Get blur parameters
        blur_kernel_size = config.get('canvas_blur_kernel_size', 0)
        blur_sigma = config.get('canvas_blur_sigma', 1.0)

        model = FullModelWithCanvasPainter(
            encoder,
            canvas_painter,
            head,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma
        )
    else:
        print("  Detected bottleneck model")

        latent_dim = latent_channels * latent_size * latent_size

        encoder = SimpleTextEncoder(
            num_tokens=len(vocab),
            max_len=12,
            d_model=128,
            nhead=4,
            ff_dim=256,
            num_layers=2,
            dropout=0.1,
            latent_dim=latent_dim
        )

        model = FullModelWithBottleneck(encoder, head, latent_size, latent_channels)

    # If using maxpool, need to do a dummy forward pass to initialize the MLP
    if use_maxpool:
        with torch.no_grad():
            dummy_tokens = torch.zeros(1, 12, dtype=torch.long).to(device)
            dummy_mask = torch.ones(1, 12, dtype=torch.long).to(device)

            if use_canvas_painter:
                # Canvas painter model
                token_reps, global_rep = model.encoder(dummy_tokens, dummy_mask, return_token_reps=True)
                dummy_canvas = model.canvas_painter(token_reps, global_rep, dummy_mask)
                _ = model.head(dummy_canvas)
            else:
                # Bottleneck model
                dummy_latent_vec = model.encoder(dummy_tokens, dummy_mask)
                dummy_latent_img = dummy_latent_vec.view(1, latent_channels, latent_size, latent_size)
                _ = model.head(dummy_latent_img)  # This initializes the MLP

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, property_encoder, config


def compute_gradient_x_input_attribution(
    model,
    Z: torch.Tensor,
    head_name: str,
    target_class_idx: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """Compute Gradient × Input attribution for a specific property head.

    Args:
        model: FullModelWithBottleneck instance.
        Z: Latent image tensor [1, 3, 32, 32] with gradients enabled.
        head_name: Name of property head ('color1', 'size1', etc.).
        target_class_idx: Index of class to compute attribution for.
        device: Device.

    Returns:
        Attribution map [3, 32, 32] showing importance per channel.
    """
    # Ensure Z requires grad
    Z = Z.detach().clone().requires_grad_(True)

    # Forward pass through CNN head only
    outputs = model.head(Z)

    # Get logits for target head
    logits = outputs[head_name]  # [1, num_classes]
    target_logit = logits[0, target_class_idx]

    # Backprop to compute gradients w.r.t. Z
    model.zero_grad()
    target_logit.backward()

    # Gradient × Input attribution
    gradients = Z.grad.detach()  # [1, 3, 32, 32]
    attribution = Z.detach() * gradients  # [1, 3, 32, 32]

    # Remove batch dimension
    attribution = attribution.squeeze(0)  # [3, 32, 32]

    return attribution


def aggregate_attribution_l2(attr_per_channel: torch.Tensor) -> torch.Tensor:
    """Aggregate per-channel attribution using L2-norm.

    Args:
        attr_per_channel: Attribution tensor [3, H, W].

    Returns:
        Combined attribution [H, W].
    """
    # L2-norm across channels: sqrt(R^2 + G^2 + B^2)
    combined = torch.sqrt((attr_per_channel ** 2).sum(dim=0))
    return combined


def normalize_saliency(saliency: torch.Tensor) -> np.ndarray:
    """Normalize saliency map to [0, 1] range for visualization.

    Args:
        saliency: Saliency tensor [H, W].

    Returns:
        Normalized numpy array [H, W].
    """
    saliency_np = saliency.cpu().numpy()

    # Handle edge case of all zeros
    if saliency_np.max() == saliency_np.min():
        return np.zeros_like(saliency_np)

    # Normalize to [0, 1]
    normalized = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min())
    return normalized


def apply_heatmap_colormap(saliency_map: np.ndarray, colormap='coolwarm') -> np.ndarray:
    """Apply heatmap colormap to saliency map.

    Args:
        saliency_map: Normalized saliency [H, W] in [0, 1].
        colormap: Matplotlib colormap name ('coolwarm', 'jet', 'hot', etc.).

    Returns:
        RGB image [H, W, 3] in [0, 1] range.
    """
    cmap = plt.colormaps.get_cmap(colormap)
    rgb_image = cmap(saliency_map)[:, :, :3]  # Drop alpha channel
    return rgb_image


def create_saliency_grid(
    latent_image: torch.Tensor,
    all_attributions: Dict[str, Dict[str, torch.Tensor]],
    property_encoder: PropertyEncoder,
    predictions: Dict[str, int],
    sentence: str,
    output_path: str
):
    """Create and save 7×5 visualization grid.

    Args:
        latent_image: Original latent image [3, 32, 32].
        all_attributions: Dict mapping head_name to {'per_channel': [3,32,32], 'combined': [32,32]}.
        property_encoder: PropertyEncoder for getting class names.
        predictions: Dict mapping head_name to predicted class index.
        output_path: Path to save output image.
    """
    # Property heads in order
    heads = ['color1', 'size1', 'shape1', 'color2', 'size2', 'shape2', 'rel']

    # Create figure with 7 rows × 5 columns
    fig, axes = plt.subplots(7, 5, figsize=(20, 28))

    # Normalize latent image for display
    latent_normalized = normalize_latent_image(latent_image.unsqueeze(0)).squeeze(0)  # [3, 32, 32]
    latent_rgb = latent_normalized.permute(1, 2, 0).cpu().numpy()  # [32, 32, 3]

    for i, head in enumerate(heads):
        attr_data = all_attributions[head]
        per_channel_attr = attr_data['per_channel']  # [3, 32, 32]
        combined_attr = attr_data['combined']  # [32, 32]

        # Get predicted class name for title
        pred_idx = predictions[head]
        if head == 'rel':
            class_name = property_encoder.id2rel[pred_idx]
        elif 'color' in head:
            class_name = property_encoder.id2color[pred_idx]
        elif 'size' in head:
            class_name = property_encoder.id2size[pred_idx]
        else:  # shape
            class_name = property_encoder.id2shape[pred_idx]

        # Column 0: Original latent image with property label
        axes[i, 0].imshow(latent_rgb)
        if i == 0:
            # First row: include column header
            axes[i, 0].set_title(f'Latent\n{head} (pred: {class_name})', fontsize=10, fontweight='bold')
        else:
            # Other rows: just property and prediction
            axes[i, 0].set_title(f'{head}\n(pred: {class_name})', fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # Columns 1-3: Per-channel attributions (R, G, B)
        channel_names = ['R', 'G', 'B']
        for c in range(3):
            saliency_norm = normalize_saliency(per_channel_attr[c])
            saliency_colored = apply_heatmap_colormap(saliency_norm, 'coolwarm')
            axes[i, c + 1].imshow(saliency_colored)
            if i == 0:  # Add column headers only on first row
                axes[i, c + 1].set_title(f'{channel_names[c]}-channel', fontsize=12, fontweight='bold')
            axes[i, c + 1].axis('off')

        # Column 4: Combined attribution
        combined_norm = normalize_saliency(combined_attr)
        combined_colored = apply_heatmap_colormap(combined_norm, 'coolwarm')
        axes[i, 4].imshow(combined_colored)
        if i == 0:
            axes[i, 4].set_title('Combined\n(L2-norm)', fontsize=12, fontweight='bold')
        axes[i, 4].axis('off')

    # Add overall title with input sentence
    fig.suptitle(f'Saliency Maps for: "{sentence}"\nWhich Latent Regions Drive Each Property Prediction?',
                 fontsize=16, fontweight='bold', y=0.998)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved saliency visualization to: {output_path}")
    plt.close()


def visualize_saliency(
    sentence: str,
    checkpoint_path: str,
    data_path: str,
    output_path: str = 'saliency_viz.png',
    device: str = 'cpu'
):
    """Main function to generate saliency visualization.

    Args:
        sentence: Input sentence string.
        checkpoint_path: Path to model checkpoint.
        data_path: Path to training data JSON (contains vocab and metadata).
        output_path: Path to save output image.
        device: Device to run on.
    """
    print(f"Loading model from {checkpoint_path}...")
    model, tokenizer, property_encoder, config = load_model(
        checkpoint_path, data_path, device
    )

    print(f"Processing sentence: '{sentence}'")

    # Tokenize sentence
    token_ids, attn_mask = tokenizer.encode(sentence)
    token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attn_mask = torch.tensor([attn_mask], dtype=torch.long, device=device)

    # Forward pass to get latent image and predictions
    with torch.no_grad():
        outputs, Z = model(token_ids, attn_mask, return_latent_image=True)

    # Get predicted classes for each head
    predictions = {}
    for head in ['color1', 'size1', 'shape1', 'color2', 'size2', 'shape2', 'rel']:
        logits = outputs[head]
        pred_class = logits.argmax(dim=1).item()
        predictions[head] = pred_class

    print("\nPredictions:")
    for head, pred_idx in predictions.items():
        if head == 'rel':
            class_name = property_encoder.id2rel[pred_idx]
        elif 'color' in head:
            class_name = property_encoder.id2color[pred_idx]
        elif 'size' in head:
            class_name = property_encoder.id2size[pred_idx]
        else:  # shape
            class_name = property_encoder.id2shape[pred_idx]
        print(f"  {head}: {class_name}")

    print("\nComputing attributions...")
    all_attributions = {}

    for head in ['color1', 'size1', 'shape1', 'color2', 'size2', 'shape2', 'rel']:
        target_class = predictions[head]

        # Compute Gradient × Input attribution
        per_channel_attr = compute_gradient_x_input_attribution(
            model, Z, head, target_class, device
        )

        # Compute combined (L2-norm) attribution
        combined_attr = aggregate_attribution_l2(per_channel_attr)

        all_attributions[head] = {
            'per_channel': per_channel_attr,
            'combined': combined_attr
        }

    print(f"\nCreating visualization grid...")
    create_saliency_grid(
        Z.squeeze(0),  # Remove batch dimension
        all_attributions,
        property_encoder,
        predictions,
        sentence,
        output_path
    )

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize saliency maps for latent bottleneck model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to training data JSON (contains vocab and metadata)'
    )
    parser.add_argument(
        '--sentence',
        type=str,
        required=True,
        help='Input sentence to analyze'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='saliency_viz.png',
        help='Output image path (default: saliency_viz.png)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to run on (default: cpu)'
    )

    args = parser.parse_args()

    visualize_saliency(
        sentence=args.sentence,
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_path=args.output,
        device=args.device
    )


if __name__ == '__main__':
    main()
