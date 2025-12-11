#!/usr/bin/env python3
"""Training script for the simple text encoder."""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple

from vocab import build_vocab_from_data, Tokenizer, Vocabulary
from model import SimpleTextEncoder, PropertyHead, FullModel, ImagePropertyHead, FullModelWithBottleneck
from dataset import PropertyEncoder, create_dataloaders
from image_utils import save_latent_images
from scene_generator import scene_to_sentence


def compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute multi-task loss across all properties.

    Args:
        outputs: Model predictions (logits).
        batch: Batch with ground truth labels.

    Returns:
        Total loss (average of all property losses).
    """
    criterion = nn.CrossEntropyLoss()

    losses = []
    property_names = ['color1', 'size1', 'shape1', 'color2', 'size2', 'shape2', 'rel']

    for prop in property_names:
        loss = criterion(outputs[prop], batch[prop])
        losses.append(loss)

    total_loss = sum(losses) / len(losses)
    return total_loss


def compute_mask_sparsity_loss(mask: torch.Tensor) -> torch.Tensor:
    """Compute sparsity loss to prevent mask from covering entire latent.

    Penalizes the mean value of the mask, encouraging sparse (low average) coverage.

    Args:
        mask: [batch_size, 1, height, width] Learned mask in range [0, 1].

    Returns:
        Scalar sparsity loss (lower = more sparse).
    """
    return mask.mean()


def compute_mask_smoothness_loss(mask: torch.Tensor) -> torch.Tensor:
    """Compute smoothness (TV-like) loss to concentrate mask in one area.

    Encourages mask to be a single contiguous region rather than scattered speckles.
    Uses isotropic L1 TV: sqrt(dx^2 + dy^2).

    Args:
        mask: [batch_size, 1, height, width] Learned mask in range [0, 1].

    Returns:
        Scalar smoothness loss (lower = more concentrated/smooth).
    """
    # Compute horizontal gradients (x direction)
    grad_x = mask[:, :, :, 1:] - mask[:, :, :, :-1]  # [B, 1, H, W-1]

    # Compute vertical gradients (y direction)
    grad_y = mask[:, :, 1:, :] - mask[:, :, :-1, :]  # [B, 1, H-1, W]

    # Crop to common spatial extent: [H-1, W-1]
    grad_x_cropped = grad_x[:, :, :-1, :]  # [B, 1, H-1, W-1]
    grad_y_cropped = grad_y[:, :, :, :-1]  # [B, 1, H-1, W-1]

    # Compute isotropic TV: sqrt(grad_x^2 + grad_y^2 + eps)
    eps = 1e-8
    tv_loss = torch.sqrt(grad_x_cropped ** 2 + grad_y_cropped ** 2 + eps).mean()

    return tv_loss


def compute_mask_binary_loss(mask: torch.Tensor) -> torch.Tensor:
    """Compute binary loss to push mask values toward 0 or 1.

    Loss is minimized when mask values are binary (either 0 or 1).
    Uses the formula: mean(mask * (1 - mask)) which equals 0 for binary values.

    Args:
        mask: [batch_size, 1, height, width] Learned mask in range [0, 1].

    Returns:
        Scalar binary loss (0 when mask is perfectly binary).
    """
    return (mask * (1.0 - mask)).mean()


def compute_accuracy(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute per-property accuracy.

    Args:
        outputs: Model predictions (logits).
        batch: Batch with ground truth labels.

    Returns:
        Dictionary of accuracy for each property.
    """
    accuracies = {}
    property_names = ['color1', 'size1', 'shape1', 'color2', 'size2', 'shape2', 'rel']

    for prop in property_names:
        preds = outputs[prop].argmax(dim=-1)
        correct = (preds == batch[prop]).sum().item()
        total = batch[prop].size(0)
        accuracies[prop] = correct / total

    return accuracies


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    noise_stddev: float = 0.0,
    classification_weight: float = 1.0,
    mask_sparsity_weight: float = 0.0,
    mask_smoothness_weight: float = 0.0,
    mask_binary_weight: float = 0.0
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Either FullModel or FullModelWithBottleneck.
        train_loader: DataLoader for training data.
        optimizer: Optimizer.
        device: Device to train on.
        noise_stddev: Standard deviation of Gaussian noise to add to latent (only applies to bottleneck models).
        classification_weight: Weight for classification loss.
        mask_sparsity_weight: Weight for mask sparsity loss (only applies when model uses mask).
        mask_smoothness_weight: Weight for mask smoothness loss (only applies when model uses mask).
        mask_binary_weight: Weight for mask binary loss (only applies when model uses mask).

    Returns:
        Dictionary with average loss, mask losses, and accuracies.
    """
    model.train()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_mask_sparsity_loss = 0.0
    total_mask_smoothness_loss = 0.0
    total_mask_binary_loss = 0.0
    total_accs = {prop: 0.0 for prop in ['color1', 'size1', 'shape1', 'color2', 'size2', 'shape2', 'rel']}
    num_batches = 0

    # Check if model uses mask
    use_mask = isinstance(model, FullModelWithBottleneck) and hasattr(model, 'use_mask') and model.use_mask

    for batch in train_loader:
        # Move to device
        token_ids = batch['token_ids'].to(device)
        attn_mask = batch['attn_mask'].to(device)
        labels = {k: v.to(device) for k, v in batch.items() if k not in ['token_ids', 'attn_mask']}

        # Forward pass
        if use_mask:
            # Need to get mask for loss computation
            outputs, Z, mask = model(token_ids, attn_mask, return_latent_image=True, noise_stddev=noise_stddev)
        elif isinstance(model, FullModelWithBottleneck):
            outputs = model(token_ids, attn_mask, noise_stddev=noise_stddev)
        else:
            outputs = model(token_ids, attn_mask)

        # Handle tuple return from FullModelWithBottleneck without explicit return_latent_image
        if isinstance(outputs, tuple) and not use_mask:
            outputs = outputs[0]

        # Compute classification loss
        classification_loss = compute_loss(outputs, labels)
        loss = classification_weight * classification_loss
        total_classification_loss += classification_loss.item()

        # Add mask losses if using mask
        if use_mask and (mask_sparsity_weight > 0 or mask_smoothness_weight > 0 or mask_binary_weight > 0):
            if mask_sparsity_weight > 0:
                sparsity_loss = compute_mask_sparsity_loss(mask)
                loss = loss + mask_sparsity_weight * sparsity_loss
                total_mask_sparsity_loss += sparsity_loss.item()

            if mask_smoothness_weight > 0:
                smoothness_loss = compute_mask_smoothness_loss(mask)
                loss = loss + mask_smoothness_weight * smoothness_loss
                total_mask_smoothness_loss += smoothness_loss.item()

            if mask_binary_weight > 0:
                binary_loss = compute_mask_binary_loss(mask)
                loss = loss + mask_binary_weight * binary_loss
                total_mask_binary_loss += binary_loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        accs = compute_accuracy(outputs, labels)
        for prop, acc in accs.items():
            total_accs[prop] += acc
        num_batches += 1

    # Average metrics
    avg_loss = total_loss / num_batches
    avg_classification_loss = total_classification_loss / num_batches
    avg_mask_sparsity = total_mask_sparsity_loss / num_batches if use_mask and mask_sparsity_weight > 0 else 0.0
    avg_mask_smoothness = total_mask_smoothness_loss / num_batches if use_mask and mask_smoothness_weight > 0 else 0.0
    avg_mask_binary = total_mask_binary_loss / num_batches if use_mask and mask_binary_weight > 0 else 0.0
    avg_accs = {prop: total_accs[prop] / num_batches for prop in total_accs}

    result = {
        'loss': avg_loss,
        'classification_loss': avg_classification_loss,
        **avg_accs
    }

    # Add mask losses to result if using mask
    if use_mask:
        result['mask_sparsity'] = avg_mask_sparsity
        result['mask_smoothness'] = avg_mask_smoothness
        result['mask_binary'] = avg_mask_binary

    return result


def save_sample_latent_images(
    model,
    data_loader,
    tokenizer,
    output_dir: Path,
    epoch: int,
    device: str,
    num_samples: int = 8
):
    """Save sample latent images from validation data.

    Args:
        model: FullModelWithBottleneck instance.
        data_loader: DataLoader to sample from.
        tokenizer: Tokenizer to decode sentences.
        output_dir: Directory to save images.
        epoch: Current epoch number.
        device: Device to run inference on.
        num_samples: Number of samples to save.
    """
    if not isinstance(model, FullModelWithBottleneck):
        return  # Only save for bottleneck models

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get one batch
    batch = next(iter(data_loader))
    token_ids = batch['token_ids'][:num_samples].to(device)
    attn_mask = batch['attn_mask'][:num_samples].to(device)

    # Get latent images
    mask = None
    with torch.no_grad():
        result = model(token_ids, attn_mask, return_latent_image=True)
        # Handle both cases: (outputs, Z) or (outputs, Z, mask)
        if len(result) == 3:
            outputs, Z, mask = result
        else:
            outputs, Z = result

    # Decode sentences
    sentences = []
    for i in range(num_samples):
        ids = token_ids[i].cpu().tolist()
        sentence = tokenizer.decode(ids, skip_special=True)
        sentences.append(sentence)

    # Save latent images
    save_latent_images(
        Z.cpu(),
        sentences,
        str(output_dir),
        prefix=f"epoch_{epoch:03d}",
        normalize=True,
        norm_method='minmax'
    )

    # Save masks if using mask module
    if mask is not None:
        mask_output_dir = output_dir / "masks"
        save_latent_images(
            mask.cpu(),
            sentences,
            str(mask_output_dir),
            prefix=f"epoch_{epoch:03d}",
            normalize=True,
            norm_method='minmax'
        )
        print(f"  → Saved {num_samples} latent images and masks to {output_dir}")
    else:
        print(f"  → Saved {num_samples} latent images to {output_dir}")


def validate(model, val_loader, device) -> Dict[str, float]:
    """Validate the model.

    Args:
        model: Either FullModel or FullModelWithBottleneck.
        val_loader: DataLoader for validation data.
        device: Device to run validation on.

    Returns:
        Dictionary with average loss and accuracies.
    """
    model.eval()
    total_loss = 0.0
    total_accs = {prop: 0.0 for prop in ['color1', 'size1', 'shape1', 'color2', 'size2', 'shape2', 'rel']}
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            token_ids = batch['token_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)
            labels = {k: v.to(device) for k, v in batch.items() if k not in ['token_ids', 'attn_mask']}

            # Forward pass
            outputs = model(token_ids, attn_mask)

            # Handle tuple return from FullModelWithBottleneck
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # outputs, Z

            # Compute loss
            loss = compute_loss(outputs, labels)

            # Track metrics
            total_loss += loss.item()
            accs = compute_accuracy(outputs, labels)
            for prop, acc in accs.items():
                total_accs[prop] += acc
            num_batches += 1

    # Average metrics
    avg_loss = total_loss / num_batches
    avg_accs = {prop: total_accs[prop] / num_batches for prop in total_accs}

    return {'loss': avg_loss, **avg_accs}


def main():
    parser = argparse.ArgumentParser(description="Train simple text encoder")
    parser.add_argument('--config', type=str, required=True, help='Path to sentence config JSON')
    parser.add_argument('--data', type=str, required=True, help='Path to generated scenes JSON')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config file if specified)')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')

    # Bottleneck architecture arguments
    parser.add_argument('--use-bottleneck', action='store_true', help='Use image bottleneck architecture')
    parser.add_argument('--latent-size', type=int, default=None, help='Spatial size of latent image (N x N, overrides config)')
    parser.add_argument('--latent-channels', type=int, default=None, help='Number of channels in latent image (1 or 3, overrides config)')
    parser.add_argument('--save-images-every', type=int, default=10, help='Save latent images every N epochs')
    parser.add_argument('--image-output-dir', type=str, default='latent_images', help='Directory for saving latent images')
    parser.add_argument('--latent-noise', type=float, default=None, help='Stddev of Gaussian noise to add to latent (overrides config)')
    parser.add_argument('--classification-weight', type=float, default=None, help='Weight for classification loss (overrides config)')
    parser.add_argument('--use-mask', action='store_true', help='Use learned mask module for spatial localization')
    parser.add_argument('--mask-sparsity-weight', type=float, default=None, help='Weight for mask sparsity loss (overrides config)')
    parser.add_argument('--mask-smoothness-weight', type=float, default=None, help='Weight for mask smoothness loss (overrides config)')
    parser.add_argument('--mask-binary-weight', type=float, default=None, help='Weight for mask binary loss (overrides config)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Build vocabulary from data file metadata
    print("Building vocabulary from data file metadata...")
    vocab = build_vocab_from_data(args.data)
    print(f"Vocabulary size: {len(vocab)}")

    # Save vocabulary
    vocab.save(output_dir / 'vocab.json')

    # Create tokenizer
    tokenizer = Tokenizer(vocab, max_len=12)

    # Create dataloaders (PropertyEncoder is created from data file metadata)
    print("Loading data...")
    train_loader, val_loader, property_encoder = create_dataloaders(
        args.data,
        tokenizer,
        batch_size=args.batch_size
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"PropertyEncoder loaded from data metadata: {property_encoder.n_colors} colors, "
          f"{property_encoder.n_sizes} sizes, {property_encoder.n_shapes} shapes, "
          f"{property_encoder.n_rels} relationships")

    # Create model
    print("Creating model...")

    if args.use_bottleneck:
        # Determine latent size (command-line overrides config)
        if args.latent_size is not None:
            latent_size = args.latent_size
            print(f"  Using latent size from command-line: {latent_size}")
        else:
            latent_size = config.get('latent_size', 32)  # Default to 32 if not in config
            print(f"  Using latent size from config: {latent_size}")

        # Determine latent channels (command-line overrides config)
        if args.latent_channels is not None:
            latent_channels = args.latent_channels
            print(f"  Using latent channels from command-line: {latent_channels}")
        else:
            latent_channels = config.get('latent_channels', 3)  # Default to 3 (RGB) if not in config
            print(f"  Using latent channels from config: {latent_channels}")

        print(f"  Image bottleneck: {latent_channels}x{latent_size}x{latent_size}")

        # Calculate latent dimension (encoder output = latent image flattened)
        latent_dim = latent_channels * latent_size * latent_size
        print(f"  Encoder output dimension: {latent_dim} (will be reshaped to image)")

        # Create encoder with latent_dim output
        encoder = SimpleTextEncoder(
            num_tokens=len(vocab),
            max_len=12,
            d_model=128,
            nhead=4,
            ff_dim=256,
            num_layers=2,
            latent_dim=latent_dim
        )

        # Get pooling configuration from config
        use_maxpool = config.get('use_maxpool', False)
        pool_size = config.get('pool_size', 2)

        if use_maxpool:
            print(f"  Using max pooling (kernel size={pool_size}) after each conv layer")
        else:
            print(f"  Using global average pooling")

        # Create CNN-based property head
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

        # Determine if using mask (command-line overrides config)
        use_mask = args.use_mask if args.use_mask else config.get('use_mask', False)
        if use_mask:
            print(f"  Using learned mask module for spatial localization")

        model = FullModelWithBottleneck(encoder, head, latent_size, latent_channels, use_mask=use_mask)
    else:
        print("  Using direct vector-to-labels architecture")

        # Create encoder with default d_model output
        encoder = SimpleTextEncoder(
            num_tokens=len(vocab),
            max_len=12,
            d_model=128,
            nhead=4,
            ff_dim=256,
            num_layers=2
        )

        # Create MLP-based property head
        head = PropertyHead(
            d_model=128,
            n_colors=property_encoder.n_colors,
            n_sizes=property_encoder.n_sizes,
            n_shapes=property_encoder.n_shapes,
            n_rels=property_encoder.n_rels
        )

        model = FullModel(encoder, head)

    model.to(args.device)

    # Determine learning rate (command-line overrides config)
    if args.lr is not None:
        learning_rate = args.lr
        print(f"  Using learning rate from command-line: {learning_rate}")
    else:
        learning_rate = config.get('learning_rate', 1e-3)  # Default to 1e-3 if not in config
        print(f"  Using learning rate from config: {learning_rate}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=args.device)

        # If using maxpool bottleneck, need to do a dummy forward pass to initialize the MLP
        if args.use_bottleneck and use_maxpool:
            print(f"  Initializing dynamic MLP for maxpool model...")
            with torch.no_grad():
                dummy_tokens = torch.zeros(1, 12, dtype=torch.long).to(args.device)
                dummy_mask = torch.ones(1, 12, dtype=torch.long).to(args.device)
                dummy_latent_vec = model.encoder(dummy_tokens, dummy_mask)
                dummy_latent_img = dummy_latent_vec.view(1, latent_channels, latent_size, latent_size)
                _ = model.head(dummy_latent_img)  # This initializes the MLP
            print(f"  ✓ MLP initialized")

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded model state")

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  ✓ Loaded optimizer state")

        # Resume from next epoch
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"  ✓ Resuming from epoch {start_epoch} (best val loss: {best_val_loss:.4f})")

        # Override config with checkpoint values if not specified on command line
        if args.use_bottleneck and 'latent_size' in checkpoint:
            if args.latent_size is None:
                latent_size = checkpoint['latent_size']
            if args.latent_channels is None:
                latent_channels = checkpoint['latent_channels']
            print(f"  ✓ Using checkpoint latent config: {latent_channels}×{latent_size}×{latent_size}")

    # Determine latent noise stddev (command-line overrides config)
    if args.latent_noise is not None:
        latent_noise_stddev = args.latent_noise
        print(f"  Using latent noise stddev from command-line: {latent_noise_stddev}")
    else:
        latent_noise_stddev = config.get('latent_noise_stddev', 0.0)  # Default to 0.0 if not in config
        if latent_noise_stddev > 0:
            print(f"  Using latent noise stddev from config: {latent_noise_stddev}")

    # Determine classification weight (command-line overrides config)
    if args.classification_weight is not None:
        classification_weight = args.classification_weight
        print(f"  Using classification weight from command-line: {classification_weight}")
    else:
        classification_weight = config.get('classification_weight', 1.0)  # Default to 1.0 if not in config
        if classification_weight != 1.0:
            print(f"  Using classification weight from config: {classification_weight}")

    # Determine mask loss weights (command-line overrides config)
    if args.mask_sparsity_weight is not None:
        mask_sparsity_weight = args.mask_sparsity_weight
        print(f"  Using mask sparsity weight from command-line: {mask_sparsity_weight}")
    else:
        mask_sparsity_weight = config.get('mask_sparsity_weight', 0.0)
        if mask_sparsity_weight > 0:
            print(f"  Using mask sparsity weight from config: {mask_sparsity_weight}")

    if args.mask_smoothness_weight is not None:
        mask_smoothness_weight = args.mask_smoothness_weight
        print(f"  Using mask smoothness weight from command-line: {mask_smoothness_weight}")
    else:
        mask_smoothness_weight = config.get('mask_smoothness_weight', 0.0)
        if mask_smoothness_weight > 0:
            print(f"  Using mask smoothness weight from config: {mask_smoothness_weight}")

    if args.mask_binary_weight is not None:
        mask_binary_weight = args.mask_binary_weight
        print(f"  Using mask binary weight from command-line: {mask_binary_weight}")
    else:
        mask_binary_weight = config.get('mask_binary_weight', 0.0)
        if mask_binary_weight > 0:
            print(f"  Using mask binary weight from config: {mask_binary_weight}")

    # Training loop
    # When resuming, --epochs means "train for N more epochs"
    # When not resuming, --epochs means "train for N epochs total"
    if args.resume:
        end_epoch = start_epoch + args.epochs
        print(f"\nResuming training on {args.device} from epoch {start_epoch} to {end_epoch - 1} ({args.epochs} additional epochs)...\n")
    else:
        end_epoch = args.epochs
        print(f"\nTraining on {args.device} for {args.epochs} epochs...\n")

    # Create image output directory if using bottleneck
    if args.use_bottleneck:
        image_output_dir = Path(args.image_output_dir)
        image_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, end_epoch):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, latent_noise_stddev,
                                    classification_weight, mask_sparsity_weight, mask_smoothness_weight, mask_binary_weight)

        # Validate
        val_metrics = validate(model, val_loader, args.device)

        # Print metrics
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Print loss breakdown if using mask
        if 'mask_sparsity' in train_metrics or 'mask_smoothness' in train_metrics or 'mask_binary' in train_metrics:
            loss_parts = [f"class: {train_metrics['classification_loss']:.4f}"]
            if 'mask_sparsity' in train_metrics and mask_sparsity_weight > 0:
                loss_parts.append(f"mask_sparse: {train_metrics['mask_sparsity']:.4f}")
            if 'mask_smoothness' in train_metrics and mask_smoothness_weight > 0:
                loss_parts.append(f"mask_smooth: {train_metrics['mask_smoothness']:.4f}")
            if 'mask_binary' in train_metrics and mask_binary_weight > 0:
                loss_parts.append(f"mask_binary: {train_metrics['mask_binary']:.4f}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} ({', '.join(loss_parts)}) | Val Loss: {val_metrics['loss']:.4f}")
        else:
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")

        print(f"  Train Acc: color1={train_metrics['color1']:.3f}, size1={train_metrics['size1']:.3f}, "
              f"shape1={train_metrics['shape1']:.3f}, color2={train_metrics['color2']:.3f}, "
              f"size2={train_metrics['size2']:.3f}, shape2={train_metrics['shape2']:.3f}, rel={train_metrics['rel']:.3f}")
        print(f"  Val Acc:   color1={val_metrics['color1']:.3f}, size1={val_metrics['size1']:.3f}, "
              f"shape1={val_metrics['shape1']:.3f}, color2={val_metrics['color2']:.3f}, "
              f"size2={val_metrics['size2']:.3f}, shape2={val_metrics['shape2']:.3f}, rel={val_metrics['rel']:.3f}")

        # Save latent images periodically
        if args.use_bottleneck and (epoch + 1) % args.save_images_every == 0:
            save_sample_latent_images(
                model, val_loader, tokenizer, image_output_dir, epoch + 1, args.device
            )

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config,
                'use_bottleneck': args.use_bottleneck,
                'latent_size': latent_size if args.use_bottleneck else None,
                'latent_channels': latent_channels if args.use_bottleneck else None,
                'use_maxpool': use_maxpool if args.use_bottleneck else None,
                'pool_size': pool_size if args.use_bottleneck else None,
                'latent_noise_stddev': latent_noise_stddev,
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

        print()

    print("Training complete!")


if __name__ == "__main__":
    main()
