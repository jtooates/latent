"""Dataset and data loading for synthetic scene sentences."""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from pathlib import Path

from vocab import Tokenizer, Vocabulary
from scene_generator import scene_to_sentence


class PropertyEncoder:
    """Encodes property values to class indices for classification."""

    def __init__(self, colors: List[str], sizes: List[str], shapes: List[str], rels: List[str]):
        """Initialize property encoders.

        Args:
            colors: List of color values.
            sizes: List of size values.
            shapes: List of shape values.
            rels: List of relationship values.
        """
        # Add NOT_MENTIONED as a special class for each property
        self.colors = ["NOT_MENTIONED"] + colors
        self.sizes = ["NOT_MENTIONED"] + sizes
        self.shapes = ["NOT_MENTIONED"] + shapes
        self.rels = ["NOT_MENTIONED"] + rels

        # Create mappings
        self.color2id = {c: i for i, c in enumerate(self.colors)}
        self.size2id = {s: i for i, s in enumerate(self.sizes)}
        self.shape2id = {s: i for i, s in enumerate(self.shapes)}
        self.rel2id = {r: i for i, r in enumerate(self.rels)}

        # Reverse mappings
        self.id2color = {i: c for c, i in self.color2id.items()}
        self.id2size = {i: s for s, i in self.size2id.items()}
        self.id2shape = {i: s for s, i in self.shape2id.items()}
        self.id2rel = {i: r for r, i in self.rel2id.items()}

    def encode_object(self, obj: Dict[str, str]) -> Tuple[int, int, int]:
        """Encode object properties to class indices.

        Args:
            obj: Object dict with 'color', 'size', 'shape'.

        Returns:
            Tuple of (color_id, size_id, shape_id).
        """
        color = obj.get("color", "none")
        size = obj.get("size", "none")
        shape = obj.get("shape", "none")

        # Map "none" to "NOT_MENTIONED"
        color_id = self.color2id.get(color if color != "none" else "NOT_MENTIONED", 0)
        size_id = self.size2id.get(size if size != "none" else "NOT_MENTIONED", 0)
        shape_id = self.shape2id.get(shape if shape != "none" else "NOT_MENTIONED", 0)

        return color_id, size_id, shape_id

    def encode_relationship(self, rel: str) -> int:
        """Encode relationship to class index.

        Args:
            rel: Relationship string.

        Returns:
            Relationship class ID.
        """
        rel_id = self.rel2id.get(rel if rel != "none" else "NOT_MENTIONED", 0)
        return rel_id

    @property
    def n_colors(self) -> int:
        return len(self.colors)

    @property
    def n_sizes(self) -> int:
        return len(self.sizes)

    @property
    def n_shapes(self) -> int:
        return len(self.shapes)

    @property
    def n_rels(self) -> int:
        return len(self.rels)


class SceneDataset(Dataset):
    """PyTorch dataset for synthetic scene sentences."""

    def __init__(
        self,
        scenes_path: str,
        tokenizer: Tokenizer,
        property_encoder: PropertyEncoder
    ):
        """Initialize dataset.

        Args:
            scenes_path: Path to JSON file with generated scenes.
            tokenizer: Tokenizer for encoding sentences.
            property_encoder: PropertyEncoder for encoding labels.
        """
        self.tokenizer = tokenizer
        self.property_encoder = property_encoder

        # Load scenes
        with open(scenes_path, 'r') as f:
            data = json.load(f)
        self.scenes = data['scenes']

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example.

        Returns:
            Dictionary with:
                - token_ids: [seq_len] Token IDs
                - attn_mask: [seq_len] Attention mask
                - color1, size1, shape1: Object 1 labels
                - color2, size2, shape2: Object 2 labels
                - rel: Relationship label
        """
        scene = self.scenes[idx]

        # Convert scene to sentence
        sentence = scene_to_sentence(scene)

        # Tokenize sentence
        token_ids, attn_mask = self.tokenizer.encode(sentence)

        # Extract objects
        objects = scene['objects']
        relationship = scene['relationship']

        # Encode object 1 (always present)
        obj1 = objects[0]
        color1_id, size1_id, shape1_id = self.property_encoder.encode_object(obj1)

        # Encode object 2 (if present, otherwise NOT_MENTIONED)
        if len(objects) > 1:
            obj2 = objects[1]
            color2_id, size2_id, shape2_id = self.property_encoder.encode_object(obj2)
        else:
            # NOT_MENTIONED is always class 0
            color2_id, size2_id, shape2_id = 0, 0, 0

        # Encode relationship
        rel_id = self.property_encoder.encode_relationship(relationship)

        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'attn_mask': torch.tensor(attn_mask, dtype=torch.long),
            'color1': torch.tensor(color1_id, dtype=torch.long),
            'size1': torch.tensor(size1_id, dtype=torch.long),
            'shape1': torch.tensor(shape1_id, dtype=torch.long),
            'color2': torch.tensor(color2_id, dtype=torch.long),
            'size2': torch.tensor(size2_id, dtype=torch.long),
            'shape2': torch.tensor(shape2_id, dtype=torch.long),
            'rel': torch.tensor(rel_id, dtype=torch.long),
        }


def create_dataloaders(
    scenes_path: str,
    tokenizer: Tokenizer,
    property_encoder: PropertyEncoder,
    batch_size: int = 32,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        scenes_path: Path to scenes JSON file.
        tokenizer: Tokenizer instance.
        property_encoder: PropertyEncoder instance.
        batch_size: Batch size.
        train_split: Fraction of data for training.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    dataset = SceneDataset(scenes_path, tokenizer, property_encoder)

    # Split into train and validation
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader
