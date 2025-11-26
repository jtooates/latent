"""Neural network models for text encoding and property prediction."""

import torch
import torch.nn as nn
from typing import Dict


class SimpleTextEncoder(nn.Module):
    """Transformer-based text encoder that maps sentences to latent vectors.

    Architecture:
        token ids -> embeddings + positional encodings
                  -> 2-layer TransformerEncoder
                  -> [CLS] hidden state
                  -> optional projection to match latent image size
    """

    def __init__(
        self,
        num_tokens: int,
        max_len: int = 12,
        d_model: int = 128,
        nhead: int = 4,
        ff_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        latent_dim: int = None
    ):
        """Initialize the text encoder.

        Args:
            num_tokens: Vocabulary size.
            max_len: Maximum sequence length.
            d_model: Embedding and hidden dimension.
            nhead: Number of attention heads.
            ff_dim: Feedforward hidden dimension.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout probability.
            latent_dim: Output latent dimension. If None, uses d_model.
                       For bottleneck models, set to latent_size * latent_size * latent_channels.
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.latent_dim = latent_dim if latent_dim is not None else d_model

        # Token and positional embeddings
        self.token_emb = nn.Embedding(num_tokens, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Optional projection to match target latent dimension
        if self.latent_dim != d_model:
            self.projection = nn.Linear(d_model, self.latent_dim)
        else:
            self.projection = None

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode token IDs to latent vector.

        Args:
            token_ids: [batch_size, seq_len] Token IDs (includes [CLS] at position 0).
            attn_mask: [batch_size, seq_len] Attention mask (1 for real tokens, 0 for PAD).

        Returns:
            latent: [batch_size, latent_dim] Latent vector from [CLS] token.
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Generate position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Embed tokens and add positional encodings
        x = self.token_emb(token_ids) + self.pos_emb(positions)  # [B, L, d_model]

        # Create padding mask for transformer (True = ignore)
        src_key_padding_mask = None
        if attn_mask is not None:
            src_key_padding_mask = (attn_mask == 0)  # [B, L], bool

        # Pass through transformer encoder
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, L, d_model]

        # Extract [CLS] token representation
        latent = h[:, 0, :]  # [B, d_model]

        # Project to target dimension if needed
        if self.projection is not None:
            latent = self.projection(latent)  # [B, latent_dim]

        return latent


class PropertyHead(nn.Module):
    """MLP classification head for predicting object properties and relationships.

    Predicts 7 properties:
        - Object 1: color, size, shape
        - Object 2: color, size, shape
        - Relationship between objects

    Each property is a multi-class classification with a NOT_MENTIONED class.
    """

    def __init__(
        self,
        d_model: int,
        n_colors: int,
        n_sizes: int,
        n_shapes: int,
        n_rels: int,
        hidden_dim: int = 128
    ):
        """Initialize the property head.

        Args:
            d_model: Input latent dimension.
            n_colors: Number of color classes (including NOT_MENTIONED).
            n_sizes: Number of size classes (including NOT_MENTIONED).
            n_shapes: Number of shape classes (including NOT_MENTIONED).
            n_rels: Number of relationship classes (including NOT_MENTIONED).
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
        )

        # Object 1 classifiers
        self.color1 = nn.Linear(hidden_dim, n_colors)
        self.size1 = nn.Linear(hidden_dim, n_sizes)
        self.shape1 = nn.Linear(hidden_dim, n_shapes)

        # Object 2 classifiers
        self.color2 = nn.Linear(hidden_dim, n_colors)
        self.size2 = nn.Linear(hidden_dim, n_sizes)
        self.shape2 = nn.Linear(hidden_dim, n_shapes)

        # Relationship classifier
        self.rel = nn.Linear(hidden_dim, n_rels)

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict properties from latent vector.

        Args:
            latent: [batch_size, d_model] Latent vectors.

        Returns:
            Dictionary of logits for each property:
                - color1, size1, shape1: Object 1 properties
                - color2, size2, shape2: Object 2 properties
                - rel: Relationship
        """
        h = self.mlp(latent)

        return {
            "color1": self.color1(h),
            "size1": self.size1(h),
            "shape1": self.shape1(h),
            "color2": self.color2(h),
            "size2": self.size2(h),
            "shape2": self.shape2(h),
            "rel": self.rel(h),
        }


class ImageBottleneck(nn.Module):
    """Maps latent vector to RGB-like image tensor.

    Architecture:
        latent_vec [B, d_model] -> MLP -> reshape -> Z [B, C, N, N]

    This bottleneck forces semantic information to be encoded in a visual
    spatial representation that can be decoded by a CNN.
    """

    def __init__(
        self,
        d_model: int,
        img_size: int = 32,
        img_channels: int = 3,
        hidden_dim: int = 256
    ):
        """Initialize the image bottleneck.

        Args:
            d_model: Input latent dimension from text encoder.
            img_size: Spatial size N of output image (N x N).
            img_channels: Number of channels C (1 for grayscale, 3 for RGB).
            hidden_dim: Hidden layer dimension in MLP.
        """
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels

        # MLP to map latent vector to flattened image
        output_dim = img_channels * img_size * img_size
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, latent_vec: torch.Tensor) -> torch.Tensor:
        """Map latent vector to image-like tensor.

        Args:
            latent_vec: [batch_size, d_model] Latent vectors from text encoder.

        Returns:
            Z: [batch_size, img_channels, img_size, img_size] Latent image.
        """
        B = latent_vec.size(0)
        x = self.mlp(latent_vec)  # [B, C*H*W]
        Z = x.view(B, self.img_channels, self.img_size, self.img_size)  # [B, C, H, W]
        return Z


class ImagePropertyHead(nn.Module):
    """CNN-based property head that reads from latent images.

    Instead of taking a latent vector, this head takes an image-like tensor
    and uses convolutions to extract features, forcing the model to encode
    semantic information spatially.

    Predicts 7 properties:
        - Object 1: color, size, shape
        - Object 2: color, size, shape
        - Relationship between objects
    """

    def __init__(
        self,
        img_channels: int,
        n_colors: int,
        n_sizes: int,
        n_shapes: int,
        n_rels: int,
        conv_channels: tuple = (32, 64),
        hidden_dim: int = 128,
        use_maxpool: bool = False,
        pool_size: int = 2
    ):
        """Initialize the image property head.

        Args:
            img_channels: Number of input channels in latent image.
            n_colors: Number of color classes (including NOT_MENTIONED).
            n_sizes: Number of size classes (including NOT_MENTIONED).
            n_shapes: Number of shape classes (including NOT_MENTIONED).
            n_rels: Number of relationship classes (including NOT_MENTIONED).
            conv_channels: Tuple of channel sizes for conv layers.
            hidden_dim: Hidden layer dimension for MLP.
            use_maxpool: If True, use max pooling after each conv layer.
            pool_size: Size of max pooling kernel (e.g., 2 for 2x2).
        """
        super().__init__()
        self.use_maxpool = use_maxpool

        # Convolutional feature extractor
        conv_layers = []
        in_channels = img_channels
        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            ])
            if use_maxpool:
                conv_layers.append(nn.MaxPool2d(kernel_size=pool_size))
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Determine the flattened size after convolutions
        if use_maxpool:
            # Calculate spatial dimensions after max pooling
            # Starting from a square image, each maxpool divides by pool_size
            # Need to calculate this dynamically based on input size
            # For now, we'll compute it in forward pass
            self.use_global_pool = False
            # Will be set dynamically based on actual tensor size
            self.flatten_size = None
        else:
            # Global average pooling: collapse spatial dimensions
            self.use_global_pool = True
            self.pool = nn.AdaptiveAvgPool2d(1)  # Output: [B, conv_channels[-1], 1, 1]

        # MLP to create hidden representation
        # Note: If using maxpool, the input size will be determined dynamically
        if not use_maxpool:
            self.mlp = nn.Sequential(
                nn.Linear(conv_channels[-1], hidden_dim),
                nn.ReLU(),
            )
        else:
            # MLP will be created after first forward pass when we know the size
            self.mlp = None
            self.hidden_dim = hidden_dim
            self.final_channels = conv_channels[-1]

        # Property classification heads
        # Object 1
        self.color1 = nn.Linear(hidden_dim, n_colors)
        self.size1 = nn.Linear(hidden_dim, n_sizes)
        self.shape1 = nn.Linear(hidden_dim, n_shapes)

        # Object 2
        self.color2 = nn.Linear(hidden_dim, n_colors)
        self.size2 = nn.Linear(hidden_dim, n_sizes)
        self.shape2 = nn.Linear(hidden_dim, n_shapes)

        # Relationship
        self.rel = nn.Linear(hidden_dim, n_rels)

    def forward(self, Z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict properties from latent image.

        Args:
            Z: [batch_size, img_channels, H, W] Latent image.

        Returns:
            Dictionary of logits for each property:
                - color1, size1, shape1: Object 1 properties
                - color2, size2, shape2: Object 2 properties
                - rel: Relationship
        """
        B = Z.size(0)

        # Extract features with CNN
        feat = self.conv(Z)  # [B, conv_channels[-1], H', W']

        # Pooling strategy
        if self.use_global_pool:
            # Global average pooling
            feat = self.pool(feat)  # [B, conv_channels[-1], 1, 1]
            feat = feat.view(B, -1)  # [B, conv_channels[-1]]
        else:
            # Flatten after max pooling layers
            feat = feat.view(B, -1)  # [B, conv_channels[-1] * H' * W']

            # Create MLP on first forward pass if needed
            if self.mlp is None:
                flatten_size = feat.size(1)
                self.mlp = nn.Sequential(
                    nn.Linear(flatten_size, self.hidden_dim),
                    nn.ReLU(),
                ).to(feat.device)

        # MLP
        h = self.mlp(feat)  # [B, hidden_dim]

        # Predict properties
        return {
            "color1": self.color1(h),
            "size1": self.size1(h),
            "shape1": self.shape1(h),
            "color2": self.color2(h),
            "size2": self.size2(h),
            "shape2": self.shape2(h),
            "rel": self.rel(h),
        }


class FullModel(nn.Module):
    """Combined encoder and property head for end-to-end training."""

    def __init__(self, encoder: SimpleTextEncoder, head: PropertyHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder and property head.

        Args:
            token_ids: [batch_size, seq_len] Token IDs.
            attn_mask: [batch_size, seq_len] Attention mask.

        Returns:
            Dictionary of property logits.
        """
        latent = self.encoder(token_ids, attn_mask)
        outputs = self.head(latent)
        return outputs


class FullModelWithBottleneck(nn.Module):
    """Combined encoder and CNN property head with image bottleneck.

    Architecture:
        sentence -> text encoder -> latent_vec [B, C*N*N]
                                       ↓
                                   reshape
                                       ↓
                            Z [B, C, N, N] (RGB latent image)
                                       ↓
                            ImagePropertyHead (CNN-based)
                                       ↓
                                   labels

    This architecture forces semantic information to pass through a spatial
    image-like bottleneck, preventing direct vector-to-label shortcuts.
    The encoder directly outputs a vector of size (C*N*N) which is reshaped
    to an image, with no intermediate projection MLP.
    """

    def __init__(
        self,
        encoder: SimpleTextEncoder,
        head: ImagePropertyHead,
        latent_size: int = 32,
        latent_channels: int = 3
    ):
        """Initialize the full model with bottleneck.

        Args:
            encoder: Text encoder that produces latent vectors of size (latent_channels * latent_size * latent_size).
            head: CNN-based property head that reads from images.
            latent_size: Spatial size of latent image (N in N×N).
            latent_channels: Number of channels in latent image (C).
        """
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.latent_size = latent_size
        self.latent_channels = latent_channels

    def forward(
        self,
        token_ids: torch.Tensor,
        attn_mask: torch.Tensor = None,
        return_latent_image: bool = False
    ):
        """Forward pass through encoder, reshape, and property head.

        Args:
            token_ids: [batch_size, seq_len] Token IDs.
            attn_mask: [batch_size, seq_len] Attention mask.
            return_latent_image: If True, return (outputs, Z). Otherwise just outputs.

        Returns:
            If return_latent_image is False:
                Dictionary of property logits.
            If return_latent_image is True:
                Tuple of (outputs dict, latent image Z).
        """
        # 1. Encode sentence to latent vector
        latent_vec = self.encoder(token_ids, attn_mask)  # [B, C*N*N]

        # 2. Reshape to RGB latent image (no projection MLP)
        B = latent_vec.size(0)
        Z = latent_vec.view(B, self.latent_channels, self.latent_size, self.latent_size)  # [B, C, N, N]

        # 3. Predict properties from image
        outputs = self.head(Z)

        if return_latent_image:
            return outputs, Z
        else:
            return outputs
