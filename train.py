#!/usr/bin/env python3
"""Training script for the simple text encoder."""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict

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


def compute_gradient_coherence_loss(Z: torch.Tensor) -> torch.Tensor:
    """Compute gradient coherence loss to encourage similar edges across RGB channels.

    This loss penalizes differences in spatial gradients between channel pairs,
    encouraging edges and features to appear at the same locations in all channels.

    Args:
        Z: Latent image tensor [batch_size, channels, height, width].
           Assumes channels=3 for RGB.

    Returns:
        Scalar coherence loss value.
    """
    if Z.size(1) != 3:
        # Only applies to 3-channel (RGB) latents
        return torch.tensor(0.0, device=Z.device)

    # Extract individual channels
    R = Z[:, 0]  # [B, H, W]
    G = Z[:, 1]
    B = Z[:, 2]

    # Compute horizontal gradients (x direction)
    grad_x_R = R[:, :, 1:] - R[:, :, :-1]  # [B, H, W-1]
    grad_x_G = G[:, :, 1:] - G[:, :, :-1]
    grad_x_B = B[:, :, 1:] - B[:, :, :-1]

    # Compute vertical gradients (y direction)
    grad_y_R = R[:, 1:, :] - R[:, :-1, :]  # [B, H-1, W]
    grad_y_G = G[:, 1:, :] - G[:, :-1, :]
    grad_y_B = B[:, 1:, :] - B[:, :-1, :]

    # Compute pairwise gradient differences (horizontal)
    diff_x_RG = (grad_x_R - grad_x_G).abs().mean()
    diff_x_RB = (grad_x_R - grad_x_B).abs().mean()
    diff_x_GB = (grad_x_G - grad_x_B).abs().mean()

    # Compute pairwise gradient differences (vertical)
    diff_y_RG = (grad_y_R - grad_y_G).abs().mean()
    diff_y_RB = (grad_y_R - grad_y_B).abs().mean()
    diff_y_GB = (grad_y_G - grad_y_B).abs().mean()

    # Total coherence loss (average of all pairwise differences)
    coherence_loss = (diff_x_RG + diff_x_RB + diff_x_GB + diff_y_RG + diff_y_RB + diff_y_GB) / 6.0

    return coherence_loss


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


def train_epoch(model, train_loader, optimizer, device, classification_weight: float = 1.0, coherence_weight: float = 0.0, noise_stddev: float = 0.0) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Either FullModel or FullModelWithBottleneck.
        train_loader: DataLoader for training data.
        optimizer: Optimizer.
        device: Device to train on.
        classification_weight: Weight for classification loss.
        coherence_weight: Weight for gradient coherence loss (only applies to bottleneck models).
        noise_stddev: Standard deviation of Gaussian noise to add to latent (only applies to bottleneck models).

    Returns:
        Dictionary with average loss, coherence loss, and accuracies.
    """
    model.train()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_coherence_loss = 0.0
    total_accs = {prop: 0.0 for prop in ['color1', 'size1', 'shape1', 'color2', 'size2', 'shape2', 'rel']}
    num_batches = 0

    use_coherence = isinstance(model, FullModelWithBottleneck) and coherence_weight > 0

    for batch in train_loader:
        # Move to device
        token_ids = batch['token_ids'].to(device)
        attn_mask = batch['attn_mask'].to(device)
        labels = {k: v.to(device) for k, v in batch.items() if k not in ['token_ids', 'attn_mask']}

        # Forward pass
        if use_coherence:
            # Need to get latent image for coherence loss
            outputs, Z = model(token_ids, attn_mask, return_latent_image=True, noise_stddev=noise_stddev)
        else:
            if isinstance(model, FullModelWithBottleneck):
                outputs = model(token_ids, attn_mask, noise_stddev=noise_stddev)
            else:
                outputs = model(token_ids, attn_mask)
            # Handle tuple return from FullModelWithBottleneck without explicit return_latent_image
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        # Compute classification loss
        classification_loss = compute_loss(outputs, labels)

        # Compute coherence loss if applicable
        if use_coherence:
            coherence_loss = compute_gradient_coherence_loss(Z)
            loss = classification_weight * classification_loss + coherence_weight * coherence_loss
            total_coherence_loss += coherence_loss.item()
        else:
            loss = classification_weight * classification_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_classification_loss += classification_loss.item()
        accs = compute_accuracy(outputs, labels)
        for prop, acc in accs.items():
            total_accs[prop] += acc
        num_batches += 1

    # Average metrics
    avg_loss = total_loss / num_batches
    avg_classification_loss = total_classification_loss / num_batches
    avg_coherence_loss = total_coherence_loss / num_batches if use_coherence else 0.0
    avg_accs = {prop: total_accs[prop] / num_batches for prop in total_accs}

    return {
        'loss': avg_loss,
        'classification_loss': avg_classification_loss,
        'coherence_loss': avg_coherence_loss,
        **avg_accs
    }


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
    with torch.no_grad():
        outputs, Z = model(token_ids, attn_mask, return_latent_image=True)

    # Decode sentences
    sentences = []
    for i in range(num_samples):
        ids = token_ids[i].cpu().tolist()
        sentence = tokenizer.decode(ids, skip_special=True)
        sentences.append(sentence)

    # Save images
    save_latent_images(
        Z.cpu(),
        sentences,
        str(output_dir),
        prefix=f"epoch_{epoch:03d}",
        normalize=True,
        norm_method='minmax'
    )

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

    # Bottleneck architecture arguments
    parser.add_argument('--use-bottleneck', action='store_true', help='Use image bottleneck architecture')
    parser.add_argument('--latent-size', type=int, default=None, help='Spatial size of latent image (N x N, overrides config)')
    parser.add_argument('--latent-channels', type=int, default=None, help='Number of channels in latent image (1 or 3, overrides config)')
    parser.add_argument('--save-images-every', type=int, default=10, help='Save latent images every N epochs')
    parser.add_argument('--image-output-dir', type=str, default='latent_images', help='Directory for saving latent images')
    parser.add_argument('--classification-weight', type=float, default=None, help='Weight for classification loss (overrides config)')
    parser.add_argument('--coherence-weight', type=float, default=None, help='Weight for gradient coherence loss (overrides config)')
    parser.add_argument('--latent-noise', type=float, default=None, help='Stddev of Gaussian noise to add to latent (overrides config)')

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

        model = FullModelWithBottleneck(encoder, head, latent_size, latent_channels)
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

    # Determine classification loss weight (command-line overrides config)
    if args.classification_weight is not None:
        classification_weight = args.classification_weight
        print(f"  Using classification loss weight from command-line: {classification_weight}")
    else:
        classification_weight = config.get('classification_loss_weight', 1.0)  # Default to 1.0 if not in config
        if classification_weight != 1.0:
            print(f"  Using classification loss weight from config: {classification_weight}")

    # Determine coherence loss weight (command-line overrides config)
    if args.coherence_weight is not None:
        coherence_weight = args.coherence_weight
        print(f"  Using coherence loss weight from command-line: {coherence_weight}")
    else:
        coherence_weight = config.get('coherence_loss_weight', 0.0)  # Default to 0.0 if not in config
        if coherence_weight > 0:
            print(f"  Using coherence loss weight from config: {coherence_weight}")

    # Determine latent noise stddev (command-line overrides config)
    if args.latent_noise is not None:
        latent_noise_stddev = args.latent_noise
        print(f"  Using latent noise stddev from command-line: {latent_noise_stddev}")
    else:
        latent_noise_stddev = config.get('latent_noise_stddev', 0.0)  # Default to 0.0 if not in config
        if latent_noise_stddev > 0:
            print(f"  Using latent noise stddev from config: {latent_noise_stddev}")

    # Training loop
    print(f"\nTraining on {args.device} for {args.epochs} epochs...\n")
    best_val_loss = float('inf')

    # Create image output directory if using bottleneck
    if args.use_bottleneck:
        image_output_dir = Path(args.image_output_dir)
        image_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, classification_weight, coherence_weight, latent_noise_stddev)

        # Validate
        val_metrics = validate(model, val_loader, args.device)

        # Print metrics
        print(f"Epoch {epoch + 1}/{args.epochs}")
        if coherence_weight > 0:
            print(f"  Train Loss: {train_metrics['loss']:.4f} (class: {train_metrics['classification_loss']:.4f}, "
                  f"coherence: {train_metrics['coherence_loss']:.4f}) | Val Loss: {val_metrics['loss']:.4f}")
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
                'classification_loss_weight': classification_weight,
                'coherence_loss_weight': coherence_weight,
                'latent_noise_stddev': latent_noise_stddev,
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

        print()

    print("Training complete!")


if __name__ == "__main__":
    main()
