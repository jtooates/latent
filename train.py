#!/usr/bin/env python3
"""Training script for the simple text encoder."""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict

from vocab import build_vocab_from_config, Tokenizer, Vocabulary
from model import SimpleTextEncoder, PropertyHead, FullModel, ImageBottleneck, ImagePropertyHead, FullModelWithBottleneck
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


def train_epoch(model, train_loader, optimizer, device) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Either FullModel or FullModelWithBottleneck.
        train_loader: DataLoader for training data.
        optimizer: Optimizer.
        device: Device to train on.

    Returns:
        Dictionary with average loss and accuracies.
    """
    model.train()
    total_loss = 0.0
    total_accs = {prop: 0.0 for prop in ['color1', 'size1', 'shape1', 'color2', 'size2', 'shape2', 'rel']}
    num_batches = 0

    for batch in train_loader:
        # Move to device
        token_ids = batch['token_ids'].to(device)
        attn_mask = batch['attn_mask'].to(device)
        labels = {k: v.to(device) for k, v in batch.items() if k not in ['token_ids', 'attn_mask']}

        # Forward pass (works for both FullModel and FullModelWithBottleneck)
        outputs = model(token_ids, attn_mask)

        # Handle tuple return from FullModelWithBottleneck
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # outputs, Z

        # Compute loss
        loss = compute_loss(outputs, labels)

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
    avg_accs = {prop: total_accs[prop] / num_batches for prop in total_accs}

    return {'loss': avg_loss, **avg_accs}


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

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab_from_config(args.config)
    print(f"Vocabulary size: {len(vocab)}")

    # Save vocabulary
    vocab.save(output_dir / 'vocab.json')

    # Create tokenizer
    tokenizer = Tokenizer(vocab, max_len=12)

    # Create property encoder
    property_encoder = PropertyEncoder(
        colors=config['colors'],
        sizes=config['sizes'],
        shapes=config['shapes'],
        rels=config['rels']
    )

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        args.data,
        tokenizer,
        property_encoder,
        batch_size=args.batch_size
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    print("Creating model...")
    encoder = SimpleTextEncoder(
        num_tokens=len(vocab),
        max_len=12,
        d_model=128,
        nhead=4,
        ff_dim=256,
        num_layers=2
    )

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

        # Create image bottleneck
        bottleneck = ImageBottleneck(
            d_model=128,
            img_size=latent_size,
            img_channels=latent_channels,
            hidden_dim=256
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

        model = FullModelWithBottleneck(encoder, bottleneck, head)
    else:
        print("  Using direct vector-to-labels architecture")

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

    # Training loop
    print(f"\nTraining on {args.device} for {args.epochs} epochs...\n")
    best_val_loss = float('inf')

    # Create image output directory if using bottleneck
    if args.use_bottleneck:
        image_output_dir = Path(args.image_output_dir)
        image_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args.device)

        # Validate
        val_metrics = validate(model, val_loader, args.device)

        # Print metrics
        print(f"Epoch {epoch + 1}/{args.epochs}")
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
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

        print()

    print("Training complete!")


if __name__ == "__main__":
    main()
