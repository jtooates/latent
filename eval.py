#!/usr/bin/env python3
"""Evaluation and inference utilities for the simple text encoder."""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, List

from vocab import Vocabulary, Tokenizer
from model import SimpleTextEncoder, PropertyHead, FullModel, ImagePropertyHead, FullModelWithBottleneck
from dataset import PropertyEncoder
from image_utils import save_latent_image


def load_model(checkpoint_path: str, vocab_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file).
        vocab_path: Path to vocabulary JSON.
        device: Device to load model on.

    Returns:
        Tuple of (model, vocab, property_encoder, config).
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Load vocabulary
    vocab = Vocabulary.load(vocab_path)

    # Create property encoder
    property_encoder = PropertyEncoder(
        colors=config['colors'],
        sizes=config['sizes'],
        shapes=config['shapes'],
        rels=config['rels']
    )

    # Check if this is a bottleneck model
    use_bottleneck = checkpoint.get('use_bottleneck', False)

    if use_bottleneck:
        latent_size = checkpoint.get('latent_size', 32)
        latent_channels = checkpoint.get('latent_channels', 3)
        use_maxpool = checkpoint.get('use_maxpool', False)
        pool_size = checkpoint.get('pool_size', 2)

        # Calculate latent dimension
        latent_dim = latent_channels * latent_size * latent_size

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

        # Create CNN-based head
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
        # Create encoder with default d_model output
        encoder = SimpleTextEncoder(
            num_tokens=len(vocab),
            max_len=12,
            d_model=128,
            nhead=4,
            ff_dim=256,
            num_layers=2
        )

        # Create MLP-based head
        head = PropertyHead(
            d_model=128,
            n_colors=property_encoder.n_colors,
            n_sizes=property_encoder.n_sizes,
            n_shapes=property_encoder.n_shapes,
            n_rels=property_encoder.n_rels
        )

        model = FullModel(encoder, head)

    # If using maxpool, need to do a dummy forward pass to initialize the MLP
    if use_bottleneck and use_maxpool:
        with torch.no_grad():
            dummy_tokens = torch.zeros(1, 12, dtype=torch.long).to(device)
            dummy_mask = torch.ones(1, 12, dtype=torch.long).to(device)
            dummy_latent_vec = model.encoder(dummy_tokens, dummy_mask)
            dummy_latent_img = dummy_latent_vec.view(1, latent_channels, latent_size, latent_size)
            _ = model.head(dummy_latent_img)  # This initializes the MLP

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, vocab, property_encoder, config


def predict_sentence(
    sentence: str,
    model,
    tokenizer: Tokenizer,
    property_encoder: PropertyEncoder,
    device: str = 'cpu',
    return_latent_image: bool = False
):
    """Predict properties for a single sentence.

    Args:
        sentence: Input sentence string.
        model: Trained model (FullModel or FullModelWithBottleneck).
        tokenizer: Tokenizer instance.
        property_encoder: PropertyEncoder for decoding labels.
        device: Device to run inference on.
        return_latent_image: If True and model is bottleneck, return latent image.

    Returns:
        If return_latent_image is False:
            Dictionary with predicted properties.
        If return_latent_image is True and model is bottleneck:
            Tuple of (predictions dict, latent image tensor).
    """
    # Tokenize
    token_ids, attn_mask = tokenizer.encode(sentence)
    token_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    attn_mask = torch.tensor([attn_mask], dtype=torch.long).to(device)

    # Predict
    with torch.no_grad():
        if isinstance(model, FullModelWithBottleneck) and return_latent_image:
            outputs, Z = model(token_ids, attn_mask, return_latent_image=True)
        else:
            outputs = model(token_ids, attn_mask)
            if isinstance(outputs, tuple):  # Handle bottleneck without explicit return
                outputs = outputs[0]
            Z = None

    # Decode predictions
    predictions = {}
    property_names = ['color1', 'size1', 'shape1', 'color2', 'size2', 'shape2', 'rel']

    for prop in property_names:
        pred_id = outputs[prop].argmax(dim=-1).item()

        if prop.startswith('color'):
            predictions[prop] = property_encoder.id2color[pred_id]
        elif prop.startswith('size'):
            predictions[prop] = property_encoder.id2size[pred_id]
        elif prop.startswith('shape'):
            predictions[prop] = property_encoder.id2shape[pred_id]
        elif prop == 'rel':
            predictions[prop] = property_encoder.id2rel[pred_id]

    if return_latent_image and Z is not None:
        return predictions, Z
    return predictions


def format_predictions(predictions: Dict[str, str]) -> str:
    """Format predictions in a readable way.

    Args:
        predictions: Dictionary of predicted properties.

    Returns:
        Formatted string.
    """
    obj1 = []
    if predictions['color1'] != 'NOT_MENTIONED':
        obj1.append(predictions['color1'])
    if predictions['size1'] != 'NOT_MENTIONED':
        obj1.append(predictions['size1'])
    if predictions['shape1'] != 'NOT_MENTIONED':
        obj1.append(predictions['shape1'])

    obj2 = []
    if predictions['color2'] != 'NOT_MENTIONED':
        obj2.append(predictions['color2'])
    if predictions['size2'] != 'NOT_MENTIONED':
        obj2.append(predictions['size2'])
    if predictions['shape2'] != 'NOT_MENTIONED':
        obj2.append(predictions['shape2'])

    rel = predictions['rel'] if predictions['rel'] != 'NOT_MENTIONED' else None

    # Build output
    parts = []
    if obj1:
        parts.append(f"Object 1: {' '.join(obj1)}")
    if obj2:
        parts.append(f"Object 2: {' '.join(obj2)}")
    if rel:
        parts.append(f"Relationship: {rel}")

    return '\n'.join(parts) if parts else "No objects detected"


def evaluate_dataset(
    data_path: str,
    model: FullModel,
    tokenizer: Tokenizer,
    property_encoder: PropertyEncoder,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        data_path: Path to scenes JSON file.
        model: Trained model.
        tokenizer: Tokenizer instance.
        property_encoder: PropertyEncoder instance.
        device: Device to run inference on.

    Returns:
        Dictionary of accuracy metrics.
    """
    from dataset import SceneDataset
    from torch.utils.data import DataLoader

    dataset = SceneDataset(data_path, tokenizer, property_encoder)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    property_names = ['color1', 'size1', 'shape1', 'color2', 'size2', 'shape2', 'rel']
    total_correct = {prop: 0 for prop in property_names}
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            token_ids = batch['token_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)
            labels = {k: v.to(device) for k, v in batch.items() if k not in ['token_ids', 'attn_mask']}

            outputs = model(token_ids, attn_mask)
            if isinstance(outputs, tuple):  # Handle bottleneck model
                outputs = outputs[0]

            for prop in property_names:
                preds = outputs[prop].argmax(dim=-1)
                correct = (preds == labels[prop]).sum().item()
                total_correct[prop] += correct

            total_samples += token_ids.size(0)

    # Compute accuracies
    accuracies = {prop: total_correct[prop] / total_samples for prop in property_names}
    accuracies['overall'] = sum(total_correct.values()) / (total_samples * len(property_names))

    return accuracies


def main():
    parser = argparse.ArgumentParser(description="Evaluate or run inference with trained model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True, help='Path to vocabulary JSON')
    parser.add_argument('--sentence', type=str, help='Single sentence to predict (interactive mode)')
    parser.add_argument('--data', type=str, help='Path to scenes JSON for evaluation')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--visualize', action='store_true', help='Save latent image (bottleneck models only)')
    parser.add_argument('--output-image', type=str, default='latent.png', help='Output path for latent image')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, vocab, property_encoder, config = load_model(args.checkpoint, args.vocab, args.device)
    tokenizer = Tokenizer(vocab, max_len=12)
    is_bottleneck = isinstance(model, FullModelWithBottleneck)
    print(f"Model loaded successfully! (Architecture: {'Bottleneck' if is_bottleneck else 'Direct'})")

    # Single sentence prediction
    if args.sentence:
        print(f"\nInput: {args.sentence}")

        # Get predictions and optionally latent image
        if args.visualize and is_bottleneck:
            predictions, Z = predict_sentence(
                args.sentence, model, tokenizer, property_encoder, args.device, return_latent_image=True
            )

            # Save latent image
            save_latent_image(Z[0].cpu(), args.output_image, normalize=True, norm_method='minmax')
            print(f"\n  → Saved latent image to {args.output_image}")
            print(f"  → Latent image shape: {list(Z.shape[1:])}")
        else:
            predictions = predict_sentence(args.sentence, model, tokenizer, property_encoder, args.device)

        print("\nPredictions:")
        print(format_predictions(predictions))
        print("\nRaw predictions:", predictions)

    # Dataset evaluation
    elif args.data:
        print(f"\nEvaluating on dataset: {args.data}")
        accuracies = evaluate_dataset(args.data, model, tokenizer, property_encoder, args.device)
        print("\nAccuracies:")
        for prop, acc in accuracies.items():
            print(f"  {prop}: {acc:.4f}")

    else:
        print("Please provide either --sentence or --data argument.")


if __name__ == "__main__":
    main()
