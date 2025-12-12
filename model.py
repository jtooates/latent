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

    def forward(
        self,
        token_ids: torch.Tensor,
        attn_mask: torch.Tensor = None,
        return_token_reps: bool = False
    ):
        """Encode token IDs to latent vector.

        Args:
            token_ids: [batch_size, seq_len] Token IDs (includes [CLS] at position 0).
            attn_mask: [batch_size, seq_len] Attention mask (1 for real tokens, 0 for PAD).
            return_token_reps: If True, return (token_reps, global_rep).
                              If False, return just the projected latent vector.

        Returns:
            If return_token_reps is False:
                latent: [batch_size, latent_dim] Latent vector from [CLS] token.
            If return_token_reps is True:
                tuple of (token_reps, global_rep) where:
                    token_reps: [batch_size, seq_len, d_model] All token hidden states.
                    global_rep: [batch_size, d_model] Global representation from [CLS] token.
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

        if return_token_reps:
            # Return all token representations and global [CLS] representation
            token_reps = h  # [B, L, d_model]
            global_rep = h[:, 0, :]  # [B, d_model]
            return token_reps, global_rep
        else:
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


class CanvasPainterEncoder(nn.Module):
    """DRAW-style recurrent encoder that iteratively paints to a canvas latent.

    Architecture:
        - Runs for T steps where T = num_steps (if > 0) or number of tokens (if num_steps = 0)
        - At each step:
          1. Attends over all token representations
          2. Updates painter GRU state
          3. Generates a K×K write patch
          4. Writes patch to local window on canvas at learned location

    Uses local-window write instead of full-canvas Gaussian masking for tighter localization.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        H: int,
        W: int,
        C: int,
        K: int = 5,
        num_steps: int = 0
    ):
        """Initialize the canvas painter encoder.

        Args:
            d_model: Hidden size from TextEncoder.
            d_state: Hidden size of painter GRU.
            H: Canvas height.
            W: Canvas width.
            C: Canvas channels.
            K: Patch size for writing (K × K).
            num_steps: Number of painting steps. If 0, uses number of tokens.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.H = H
        self.W = W
        self.C = C
        self.K = K
        self.num_steps = num_steps

        # Attention dimension (use d_model for simplicity)
        self.d_att = d_model

        # Initialize painter state from global representation
        self.init_fc = nn.Linear(d_model, d_state)

        # Attention mechanism over tokens
        self.W_q = nn.Linear(d_state, self.d_att)
        self.W_k = nn.Linear(d_model, self.d_att)
        self.W_v = nn.Linear(d_model, self.d_att)

        # GRU for updating painter state
        # Input: [r_t, global_rep]
        gru_input_dim = self.d_att + d_model
        self.gru_cell = nn.GRUCell(gru_input_dim, d_state)

        # Write patch generation
        # Input: [s_{t+1}, r_t]
        write_input_dim = d_state + self.d_att
        self.write_fc = nn.Linear(write_input_dim, C * K * K)

        # Write location prediction
        # Input: [s_{t+1}, r_t, global_rep]
        # Output: (x_center, y_center, log_scale)
        loc_input_dim = d_state + self.d_att + d_model
        self.loc_fc = nn.Linear(loc_input_dim, 3)

    def create_gaussian_mask(
        self,
        center_x: torch.Tensor,
        center_y: torch.Tensor,
        scale: torch.Tensor
    ) -> torch.Tensor:
        """Create 2D Gaussian mask centered at (center_x, center_y).

        Args:
            center_x: [batch_size] x-coordinates in range [-1, 1].
            center_y: [batch_size] y-coordinates in range [-1, 1].
            scale: [batch_size] scale factor (larger = more spread).

        Returns:
            mask: [batch_size, 1, H, W] Gaussian mask.
        """
        B = center_x.size(0)
        device = center_x.device

        # Create meshgrid for canvas coordinates in [-1, 1]
        y_coords = torch.linspace(-1, 1, self.H, device=device)
        x_coords = torch.linspace(-1, 1, self.W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        # [H, W] -> [1, H, W]
        grid_x = grid_x.unsqueeze(0)
        grid_y = grid_y.unsqueeze(0)

        # Expand centers and scale for broadcasting
        center_x = center_x.view(B, 1, 1)  # [B, 1, 1]
        center_y = center_y.view(B, 1, 1)  # [B, 1, 1]
        scale = scale.view(B, 1, 1)  # [B, 1, 1]

        # Compute squared distance from center
        dist_sq = ((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)

        # Gaussian with variance controlled by scale
        # Using exp(-dist^2 / (2 * sigma^2)) where sigma = scale
        variance = scale ** 2 + 1e-8
        mask = torch.exp(-dist_sq / (2 * variance))

        # Add channel dimension: [B, H, W] -> [B, 1, H, W]
        mask = mask.unsqueeze(1)

        return mask

    def _normalized_to_indices(
        self,
        x_norm: torch.Tensor,  # [B] or [B, 1] in [-1, 1] for width
        y_norm: torch.Tensor,  # [B] or [B, 1] in [-1, 1] for height
        W: int,
        H: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert normalized coords to integer canvas indices.

        Args:
            x_norm: Tensor [B] or [B, 1] in [-1, 1] for width
            y_norm: Tensor [B] or [B, 1] in [-1, 1] for height
            W: canvas width
            H: canvas height

        Returns:
            j_indices: LongTensor [B] in [0, W-1]
            i_indices: LongTensor [B] in [0, H-1]
        """
        # Ensure shape is [B, 1] for computation
        if x_norm.dim() == 1:
            x_norm = x_norm.unsqueeze(-1)
        if y_norm.dim() == 1:
            y_norm = y_norm.unsqueeze(-1)

        # Map [-1, 1] -> [0, W-1] and [0, H-1]
        j = 0.5 * (x_norm + 1.0) * (W - 1)
        i = 0.5 * (y_norm + 1.0) * (H - 1)

        # Round to nearest integer and clamp to valid range
        j = j.round().long().clamp(0, W - 1)  # [B, 1]
        i = i.round().long().clamp(0, H - 1)  # [B, 1]

        # Squeeze to [B]
        return j.squeeze(-1), i.squeeze(-1)

    def _write_patch_to_window(
        self,
        canvas: torch.Tensor,    # [B, C, H, W]
        patch: torch.Tensor,     # [B, C, K, K]
        j_indices: torch.Tensor, # [B] - width/column centers
        i_indices: torch.Tensor  # [B] - height/row centers
    ) -> torch.Tensor:
        """Write patches to local windows on the canvas.

        Args:
            canvas: [B, C, H, W] Canvas tensor
            patch: [B, C, K, K] Patch to write
            j_indices: [B] Integer column indices (width)
            i_indices: [B] Integer row indices (height)

        Returns:
            canvas: [B, C, H, W] Updated canvas with patches written
        """
        B, C, H, W = canvas.shape
        K = patch.shape[2]  # Patch size

        half_h = K // 2
        half_w = K // 2

        # Loop over batch dimension
        for b in range(B):
            j_c = j_indices[b].item()
            i_c = i_indices[b].item()

            # Compute window bounds
            top = i_c - half_h
            left = j_c - half_w
            bottom = top + K
            right = left + K

            # Clamp window to valid canvas bounds
            top_clamped = max(0, top)
            left_clamped = max(0, left)
            bottom_clamped = min(H, bottom)
            right_clamped = min(W, right)

            win_h = bottom_clamped - top_clamped
            win_w = right_clamped - left_clamped

            # Skip if window is completely out of bounds
            if win_h <= 0 or win_w <= 0:
                continue

            # Compute corresponding patch slice indices
            patch_top = top_clamped - top
            patch_left = left_clamped - left
            patch_bottom = patch_top + win_h
            patch_right = patch_left + win_w

            # Extract patch slice
            patch_slice = patch[b, :, patch_top:patch_bottom, patch_left:patch_right]  # [C, win_h, win_w]

            # Write to canvas (additive)
            canvas[b, :, top_clamped:bottom_clamped, left_clamped:right_clamped] += patch_slice

        return canvas

    def attend_to_tokens(
        self,
        s_t: torch.Tensor,
        token_reps: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute attention-weighted summary of token representations.

        Args:
            s_t: [batch_size, d_state] Current painter state.
            token_reps: [batch_size, N, d_model] Token representations.
            attention_mask: [batch_size, N] Mask (1 for valid, 0 for padding).

        Returns:
            r_t: [batch_size, d_att] Attended text summary.
        """
        # Query from state
        q_t = self.W_q(s_t)  # [B, d_att]

        # Keys and values from tokens
        k = self.W_k(token_reps)  # [B, N, d_att]
        v = self.W_v(token_reps)  # [B, N, d_att]

        # Compute attention scores
        scores = (q_t.unsqueeze(1) * k).sum(dim=-1)  # [B, N]

        # Mask out padding tokens
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Attention weights
        alpha = torch.softmax(scores, dim=-1)  # [B, N]

        # Weighted sum
        r_t = (alpha.unsqueeze(-1) * v).sum(dim=1)  # [B, d_att]

        return r_t

    def forward(
        self,
        token_reps: torch.Tensor,
        global_rep: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Paint to canvas over T steps.

        Args:
            token_reps: [batch_size, N, d_model] Token representations.
            global_rep: [batch_size, d_model] Global sentence representation.
            attention_mask: [batch_size, N] Optional mask for padding.

        Returns:
            canvas: [batch_size, C, H, W] Final painted canvas.
        """
        B, N, _ = token_reps.shape
        device = token_reps.device

        # Initialize canvas to zeros
        canvas = torch.zeros(B, self.C, self.H, self.W, device=device)

        # Initialize painter state from global representation
        s_t = torch.tanh(self.init_fc(global_rep))  # [B, d_state]

        # Determine number of steps: use num_steps if > 0, otherwise use N
        T = self.num_steps if self.num_steps > 0 else N

        # Run for T steps
        for t in range(T):
            # Step 1: Attend over all tokens
            r_t = self.attend_to_tokens(s_t, token_reps, attention_mask)  # [B, d_att]

            # Step 2: Update painter state via GRU
            # Input: [r_t, global_rep]
            gru_input = torch.cat([r_t, global_rep], dim=-1)  # [B, d_att + d_model]
            s_t = self.gru_cell(gru_input, s_t)  # [B, d_state]

            # Step 3: Generate write patch
            # Input: [s_t, r_t]
            write_input = torch.cat([s_t, r_t], dim=-1)  # [B, d_state + d_att]
            patch_vec = self.write_fc(write_input)  # [B, C * K * K]
            patch = patch_vec.view(B, self.C, self.K, self.K)  # [B, C, K, K]

            # Step 4: Predict write location
            # Input: [s_t, r_t, global_rep]
            loc_input = torch.cat([s_t, r_t, global_rep], dim=-1)  # [B, loc_input_dim]
            loc_params = self.loc_fc(loc_input)  # [B, 3]

            # Parse location parameters
            center_x = torch.tanh(loc_params[:, 0])  # [-1, 1]
            center_y = torch.tanh(loc_params[:, 1])  # [-1, 1]
            log_scale = loc_params[:, 2]
            scale = torch.exp(log_scale)  # (0, inf)
            # Clamp scale to reasonable range
            scale = torch.clamp(scale, 0.1, 2.0)

            # Step 5: Write to canvas using local-window write
            # Convert normalized centers to integer indices
            j_indices, i_indices = self._normalized_to_indices(
                center_x, center_y, self.W, self.H
            )  # [B], [B]

            # Write patch to local window on canvas
            canvas = self._write_patch_to_window(
                canvas, patch, j_indices, i_indices
            )  # [B, C, H, W]

        return canvas


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
        return_latent_image: bool = False,
        noise_stddev: float = 0.0
    ):
        """Forward pass through encoder, reshape, and property head.

        Args:
            token_ids: [batch_size, seq_len] Token IDs.
            attn_mask: [batch_size, seq_len] Attention mask.
            return_latent_image: If True, return (outputs, Z). Otherwise just outputs.
            noise_stddev: Standard deviation of Gaussian noise to add to latent image.
                         Only applied during training if > 0.

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

        # 3. Add Gaussian noise during training (for regularization)
        if self.training and noise_stddev > 0:
            noise = torch.randn_like(Z) * noise_stddev
            Z_noisy = Z + noise
        else:
            Z_noisy = Z

        # 4. Predict properties from image
        outputs = self.head(Z_noisy)

        if return_latent_image:
            # Return clean latent image (without noise)
            return outputs, Z
        else:
            return outputs


class FullModelWithCanvasPainter(nn.Module):
    """Full model using DRAW-style canvas painter encoder.

    Architecture:
        Text -> TextEncoder (gets token_reps + global_rep)
             -> CanvasPainterEncoder (iteratively paints to canvas)
             -> ImagePropertyHead (predicts properties from canvas)

    The canvas is always an image-like tensor [B, C, H, W], never a vector.
    """

    def __init__(
        self,
        encoder: SimpleTextEncoder,
        canvas_painter: CanvasPainterEncoder,
        head: ImagePropertyHead,
        blur_kernel_size: int = 0,
        blur_sigma: float = 1.0
    ):
        """Initialize the full model with canvas painter.

        Args:
            encoder: Text encoder that produces token_reps and global_rep.
            canvas_painter: DRAW-style encoder that paints to canvas.
            head: CNN-based property head that reads from canvas.
            blur_kernel_size: Size of Gaussian blur kernel (0 = no blur, 3, 5, 7, etc.).
            blur_sigma: Standard deviation for Gaussian blur.
        """
        super().__init__()
        self.encoder = encoder
        self.canvas_painter = canvas_painter
        self.head = head
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma

    def _apply_gaussian_blur(self, canvas: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to canvas to prevent high-frequency information hiding.

        Args:
            canvas: [batch_size, channels, height, width] Canvas tensor.

        Returns:
            Blurred canvas with same shape.
        """
        if self.blur_kernel_size == 0:
            return canvas

        # Simple box blur via average pooling (fast approximation)
        padding = self.blur_kernel_size // 2
        canvas_blurred = nn.functional.avg_pool2d(
            canvas,
            kernel_size=self.blur_kernel_size,
            stride=1,
            padding=padding
        )
        return canvas_blurred

    def forward(
        self,
        token_ids: torch.Tensor,
        attn_mask: torch.Tensor = None,
        return_latent_image: bool = False,
        noise_stddev: float = 0.0
    ):
        """Forward pass through encoder, canvas painter, and property head.

        Args:
            token_ids: [batch_size, seq_len] Token IDs.
            attn_mask: [batch_size, seq_len] Attention mask.
            return_latent_image: If True, return (outputs, canvas). Otherwise just outputs.
            noise_stddev: Standard deviation of Gaussian noise to add to canvas.
                         Only applied during training if > 0.

        Returns:
            If return_latent_image is False:
                Dictionary of property logits.
            If return_latent_image is True:
                Tuple of (outputs dict, canvas image).
        """
        # 1. Encode text to get token representations and global representation
        token_reps, global_rep = self.encoder(
            token_ids, attn_mask, return_token_reps=True
        )  # [B, N, d_model], [B, d_model]

        # 2. Paint to canvas using DRAW-style encoder
        canvas = self.canvas_painter(token_reps, global_rep, attn_mask)  # [B, C, H, W]

        # 3. Apply blur regularization to prevent high-frequency information hiding
        canvas = self._apply_gaussian_blur(canvas)

        # 4. Add Gaussian noise during training (for regularization)
        if self.training and noise_stddev > 0:
            noise = torch.randn_like(canvas) * noise_stddev
            canvas_noisy = canvas + noise
        else:
            canvas_noisy = canvas

        # Apply blur to noisy canvas as well
        canvas_noisy = self._apply_gaussian_blur(canvas_noisy)

        # 5. Predict properties from canvas
        outputs = self.head(canvas_noisy)

        if return_latent_image:
            # Return clean canvas (without noise)
            return outputs, canvas
        else:
            return outputs
