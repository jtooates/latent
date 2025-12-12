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
from model import (
    SimpleTextEncoder, PropertyHead, FullModel, ImagePropertyHead,
    FullModelWithBottleneck, CanvasPainterEncoder, FullModelWithCanvasPainter
)
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


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    noise_stddev: float = 0.0,
    classification_weight: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Either FullModel or FullModelWithBottleneck.
        train_loader: DataLoader for training data.
        optimizer: Optimizer.
        device: Device to train on.
        noise_stddev: Standard deviation of Gaussian noise to add to latent (only applies to bottleneck models).
        classification_weight: Weight for classification loss.

    Returns:
        Dictionary with average loss and accuracies.
    """
    model.train()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_accs = {prop: 0.0 for prop in ['color1', 'size1', 'shape1', 'color2', 'size2', 'shape2', 'rel']}
    num_batches = 0

    for batch in train_loader:
        # Move to device
        token_ids = batch['token_ids'].to(device)
        attn_mask = batch['attn_mask'].to(device)
        labels = {k: v.to(device) for k, v in batch.items() if k not in ['token_ids', 'attn_mask']}

        # Forward pass
        if isinstance(model, (FullModelWithBottleneck, FullModelWithCanvasPainter)):
            outputs = model(token_ids, attn_mask, noise_stddev=noise_stddev)
        else:
            outputs = model(token_ids, attn_mask)

        # Handle tuple return from bottleneck or canvas painter models
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Compute classification loss
        classification_loss = compute_loss(outputs, labels)
        loss = classification_weight * classification_loss
        total_classification_loss += classification_loss.item()

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
    avg_accs = {prop: total_accs[prop] / num_batches for prop in total_accs}

    return {
        'loss': avg_loss,
        'classification_loss': avg_classification_loss,
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
    if not isinstance(model, (FullModelWithBottleneck, FullModelWithCanvasPainter)):
        return  # Only save for bottleneck and canvas painter models

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

    # Save latent images (scaled to 512x512 for visibility)
    save_latent_images(
        Z.cpu(),
        sentences,
        str(output_dir),
        prefix=f"epoch_{epoch:03d}",
        normalize=True,
        norm_method='minmax',
        scale_size=512
    )

    print(f"  → Saved {num_samples} latent images (512x512) to {output_dir}")


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

    # Canvas painter arguments
    parser.add_argument('--use-canvas-painter', action='store_true', help='Use DRAW-style canvas painter encoder')
    parser.add_argument('--painter-d-state', type=int, default=None, help='Painter GRU hidden size (overrides config)')
    parser.add_argument('--painter-patch-size', type=int, default=None, help='Write patch size K (overrides config)')
    parser.add_argument('--painter-num-steps', type=int, default=None, help='Number of painting steps (0=use num tokens, overrides config)')
    parser.add_argument('--canvas-blur-kernel', type=int, default=None, help='Canvas blur kernel size (0=no blur, 3, 5, 7, etc.)')
    parser.add_argument('--canvas-blur-sigma', type=float, default=None, help='Canvas blur sigma (overrides config)')

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

    # Load checkpoint early if resuming to determine architecture
    resume_checkpoint = None
    if args.resume:
        print(f"\nDetecting architecture from checkpoint: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=args.device)
        print(f"  ✓ Checkpoint loaded (epoch {resume_checkpoint['epoch']})")

    # Create model
    print("Creating model...")

    # Check if using canvas painter (checkpoint overrides if resuming)
    if resume_checkpoint is not None:
        # When resuming, use architecture from checkpoint
        use_canvas_painter = resume_checkpoint.get('use_canvas_painter', False)
        if use_canvas_painter:
            print("  Detected canvas painter architecture from checkpoint")
        elif resume_checkpoint.get('use_bottleneck', False):
            print("  Detected bottleneck architecture from checkpoint")
        else:
            print("  Detected standard architecture from checkpoint")
    else:
        # When training from scratch, use config/CLI
        use_canvas_painter = args.use_canvas_painter if args.use_canvas_painter else config.get('use_canvas_painter', False)

    if use_canvas_painter:
        print("  Using DRAW-style canvas painter architecture")

        # When resuming, prefer checkpoint config over training_config.json
        effective_config = resume_checkpoint['config'] if resume_checkpoint else config

        # Determine latent size (command-line overrides checkpoint/config)
        if args.latent_size is not None:
            latent_size = args.latent_size
            print(f"  Using latent size from command-line: {latent_size}")
        elif resume_checkpoint and 'latent_size' in resume_checkpoint:
            latent_size = resume_checkpoint['latent_size']
            print(f"  Using latent size from checkpoint: {latent_size}")
        else:
            latent_size = effective_config.get('latent_size', 32)
            print(f"  Using latent size from config: {latent_size}")

        # Determine latent channels (command-line overrides checkpoint/config)
        if args.latent_channels is not None:
            latent_channels = args.latent_channels
            print(f"  Using latent channels from command-line: {latent_channels}")
        elif resume_checkpoint and 'latent_channels' in resume_checkpoint:
            latent_channels = resume_checkpoint['latent_channels']
            print(f"  Using latent channels from checkpoint: {latent_channels}")
        else:
            latent_channels = effective_config.get('latent_channels', 3)
            print(f"  Using latent channels from config: {latent_channels}")

        # Determine painter parameters (command-line overrides checkpoint/config)
        if args.painter_d_state is not None:
            painter_d_state = args.painter_d_state
            print(f"  Using painter d_state from command-line: {painter_d_state}")
        else:
            painter_d_state = effective_config.get('painter_d_state', 256)
            source = "checkpoint" if resume_checkpoint else "config"
            print(f"  Using painter d_state from {source}: {painter_d_state}")

        if args.painter_patch_size is not None:
            painter_patch_size = args.painter_patch_size
            print(f"  Using painter patch size from command-line: {painter_patch_size}")
        else:
            painter_patch_size = effective_config.get('painter_patch_size', 5)
            source = "checkpoint" if resume_checkpoint else "config"
            print(f"  Using painter patch size from {source}: {painter_patch_size}")

        if args.painter_num_steps is not None:
            painter_num_steps = args.painter_num_steps
            print(f"  Using painter num steps from command-line: {painter_num_steps}")
        else:
            painter_num_steps = effective_config.get('painter_num_steps', 0)
            source = "checkpoint" if resume_checkpoint else "config"
            print(f"  Using painter num steps from {source}: {painter_num_steps}")

        # Determine blur parameters (command-line overrides checkpoint/config)
        if args.canvas_blur_kernel is not None:
            canvas_blur_kernel = args.canvas_blur_kernel
            print(f"  Using canvas blur kernel from command-line: {canvas_blur_kernel}")
        else:
            canvas_blur_kernel = effective_config.get('canvas_blur_kernel_size', 0)
            source = "checkpoint" if resume_checkpoint else "config"
            print(f"  Using canvas blur kernel from {source}: {canvas_blur_kernel}")

        if args.canvas_blur_sigma is not None:
            canvas_blur_sigma = args.canvas_blur_sigma
            print(f"  Using canvas blur sigma from command-line: {canvas_blur_sigma}")
        else:
            canvas_blur_sigma = effective_config.get('canvas_blur_sigma', 1.0)
            source = "checkpoint" if resume_checkpoint else "config"
            print(f"  Using canvas blur sigma from {source}: {canvas_blur_sigma}")

        print(f"  Canvas: {latent_channels}x{latent_size}x{latent_size}")
        print(f"  Painter GRU hidden size: {painter_d_state}")
        print(f"  Write patch size: {painter_patch_size}x{painter_patch_size}")
        if painter_num_steps > 0:
            print(f"  Painting steps: {painter_num_steps} (fixed)")
        else:
            print(f"  Painting steps: adaptive (num tokens)")
        if canvas_blur_kernel > 0:
            print(f"  Canvas blur: kernel={canvas_blur_kernel}, sigma={canvas_blur_sigma}")
        else:
            print(f"  Canvas blur: disabled")

        # Create encoder (no latent_dim projection needed for canvas painter)
        encoder = SimpleTextEncoder(
            num_tokens=len(vocab),
            max_len=12,
            d_model=128,
            nhead=4,
            ff_dim=256,
            num_layers=2
        )

        # Create canvas painter
        canvas_painter = CanvasPainterEncoder(
            d_model=128,
            d_state=painter_d_state,
            H=latent_size,
            W=latent_size,
            C=latent_channels,
            K=painter_patch_size,
            num_steps=painter_num_steps
        )

        # Get pooling configuration from checkpoint or config
        if resume_checkpoint and 'use_maxpool' in resume_checkpoint:
            use_maxpool = resume_checkpoint.get('use_maxpool', False)
            pool_size = resume_checkpoint.get('pool_size', 2)
            print(f"  Using pooling config from checkpoint")
        else:
            use_maxpool = effective_config.get('use_maxpool', False)
            pool_size = effective_config.get('pool_size', 2)

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

        model = FullModelWithCanvasPainter(
            encoder,
            canvas_painter,
            head,
            blur_kernel_size=canvas_blur_kernel,
            blur_sigma=canvas_blur_sigma
        )

    elif args.use_bottleneck:
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

    # Load checkpoint weights if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_checkpoint:
        print(f"\nRestoring model state from checkpoint...")

        # If using maxpool (bottleneck or canvas painter), need dummy forward pass to initialize dynamic MLP
        if use_maxpool:
            print(f"  Initializing dynamic MLP for maxpool model...")
            with torch.no_grad():
                dummy_tokens = torch.zeros(1, 12, dtype=torch.long).to(args.device)
                dummy_mask = torch.ones(1, 12, dtype=torch.long).to(args.device)

                if use_canvas_painter:
                    # Canvas painter model
                    token_reps, global_rep = model.encoder(dummy_tokens, dummy_mask, return_token_reps=True)
                    dummy_canvas = model.canvas_painter(token_reps, global_rep, dummy_mask)
                    _ = model.head(dummy_canvas)  # This initializes the MLP
                else:
                    # Bottleneck model
                    dummy_latent_vec = model.encoder(dummy_tokens, dummy_mask)
                    dummy_latent_img = dummy_latent_vec.view(1, latent_channels, latent_size, latent_size)
                    _ = model.head(dummy_latent_img)  # This initializes the MLP
            print(f"  ✓ MLP initialized")

        # Load model state
        model.load_state_dict(resume_checkpoint['model_state_dict'])
        print(f"  ✓ Loaded model state")

        # Load optimizer state
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        print(f"  ✓ Loaded optimizer state")

        # Resume from next epoch
        start_epoch = resume_checkpoint['epoch'] + 1
        best_val_loss = resume_checkpoint.get('val_loss', float('inf'))
        print(f"  ✓ Resuming from epoch {start_epoch} (best val loss: {best_val_loss:.4f})")

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

    # Training loop
    # When resuming, --epochs means "train for N more epochs"
    # When not resuming, --epochs means "train for N epochs total"
    if args.resume:
        end_epoch = start_epoch + args.epochs
        print(f"\nResuming training on {args.device} from epoch {start_epoch} to {end_epoch - 1} ({args.epochs} additional epochs)...\n")
    else:
        end_epoch = args.epochs
        print(f"\nTraining on {args.device} for {args.epochs} epochs...\n")

    # Create image output directory if using bottleneck or canvas painter
    if args.use_bottleneck or use_canvas_painter:
        image_output_dir = Path(args.image_output_dir)
        image_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, end_epoch):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, latent_noise_stddev,
                                    classification_weight)

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
        if (args.use_bottleneck or use_canvas_painter) and (epoch + 1) % args.save_images_every == 0:
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
                'use_canvas_painter': use_canvas_painter,
                'latent_size': latent_size if (args.use_bottleneck or use_canvas_painter) else None,
                'latent_channels': latent_channels if (args.use_bottleneck or use_canvas_painter) else None,
                'use_maxpool': use_maxpool if (args.use_bottleneck or use_canvas_painter) else None,
                'pool_size': pool_size if (args.use_bottleneck or use_canvas_painter) else None,
                'latent_noise_stddev': latent_noise_stddev,
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

        print()

    print("Training complete!")


if __name__ == "__main__":
    main()
