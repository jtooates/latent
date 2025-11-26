# Simple Text Encoder for Object-Property Sentences

A PyTorch-based neural text encoder that maps synthetic sentences describing physical scenes to latent vectors. The encoder learns to extract discrete semantic properties (colors, sizes, shapes, and spatial relationships) from natural language descriptions.

## Overview

This implementation trains a small Transformer-based encoder to understand sentences like:
- "large blue circle on small red square"
- "red triangle beside green circle"
- "medium square"

The model learns to predict:
- **Object 1**: color, size, shape
- **Object 2**: color, size, shape (or NOT_MENTIONED)
- **Relationship**: spatial relationship between objects (or NOT_MENTIONED)

## Architecture

### Text Encoder (`SimpleTextEncoder`)
- **Input**: Tokenized sentence with [CLS] token prepended
- **Embeddings**: Token embeddings + positional encodings (128d)
- **Transformer**: 2-layer TransformerEncoder (4 attention heads, 256d feedforward)
- **Output**: [CLS] token representation as latent vector (128d)

### Property Head (`PropertyHead`)
- **Input**: Latent vector from encoder
- **Architecture**: Shared MLP (128d hidden) + 7 classification heads
- **Outputs**: Logits for 7 properties (color1, size1, shape1, color2, size2, shape2, rel)

### Training
- **Loss**: Multi-task cross-entropy (average of 7 property losses)
- **Optimizer**: Adam with default learning rate 1e-3
- **Data Split**: 80% train, 20% validation

## Project Structure

```
.
├── vocab.py              # Vocabulary builder and tokenizer
├── model.py              # Neural network architectures
├── dataset.py            # PyTorch dataset and data loading
├── train.py              # Training script
├── eval.py               # Evaluation and inference utilities
├── requirements.txt      # Python dependencies
├── training_config.json  # Config for generating training data (1000 scenes)
├── sentence_config.json  # Config for small dataset (10 scenes)
└── checkpoints/          # Model checkpoints and vocabulary
    ├── best_model.pt
    └── vocab.json
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- PyTorch 2.0+

## Usage

### 1. Generate Training Data

First, generate synthetic scene data using the scene generator:

```bash
# Generate 1000 training scenes
python cli.py training_config.json -o training_data.json
```

The config file specifies:
- `n`: Number of scenes to generate
- `p_one`: Probability of single-object scenes
- `p_color`, `p_size`: Probabilities for property presence
- `colors`, `sizes`, `shapes`, `rels`: Possible values for each property

### 2. Train the Model

```bash
python train.py \
  --config training_config.json \
  --data training_data.json \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-3 \
  --output-dir checkpoints \
  --device cpu
```

**Arguments:**
- `--config`: Path to sentence config JSON (for vocabulary building)
- `--data`: Path to generated scenes JSON
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate
- `--output-dir`: Directory to save checkpoints
- `--device`: Device to train on (cpu/cuda)

**Output:**
- Saves vocabulary to `checkpoints/vocab.json`
- Saves best model to `checkpoints/best_model.pt`
- Prints training/validation metrics per epoch

### 3. Inference on Single Sentences

```bash
python eval.py \
  --checkpoint checkpoints/best_model.pt \
  --vocab checkpoints/vocab.json \
  --sentence "large blue circle on small red square"
```

**Example Output:**
```
Input: large blue circle on small red square

Predictions:
Object 1: blue large circle
Object 2: red small square
Relationship: on

Raw predictions: {'color1': 'blue', 'size1': 'large', 'shape1': 'circle',
                  'color2': 'red', 'size2': 'small', 'shape2': 'square',
                  'rel': 'on'}
```

### 4. Evaluate on Dataset

```bash
python eval.py \
  --checkpoint checkpoints/best_model.pt \
  --vocab checkpoints/vocab.json \
  --data training_data.json
```

**Example Output:**
```
Accuracies:
  color1: 1.0000
  size1: 1.0000
  shape1: 1.0000
  color2: 1.0000
  size2: 1.0000
  shape2: 1.0000
  rel: 1.0000
  overall: 1.0000
```

## Training Results

### Test Training (10 epochs, 1000 scenes)

| Epoch | Train Loss | Val Loss | Val Accuracy (color1/size1/shape1/rel) |
|-------|------------|----------|----------------------------------------|
| 1     | 1.0670     | 0.8535   | 0.513 / 0.478 / 0.661 / 0.808         |
| 5     | 0.1162     | 0.0969   | 0.955 / 0.973 / 0.969 / 1.000         |
| 10    | 0.0189     | 0.0110   | 1.000 / 1.000 / 1.000 / 1.000         |

The model achieves **100% accuracy on all properties** after just 10 epochs, demonstrating that:
- The Transformer encoder successfully learns meaningful sentence-level representations
- The latent vector captures all discrete semantic attributes
- The synthetic language is learnable with a small model

## Hyperparameters

Default values (from specification):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 128 | Embedding and hidden dimension |
| `num_layers` | 2 | Transformer encoder layers |
| `nhead` | 4 | Number of attention heads |
| `ff_dim` | 256 | Feedforward hidden dimension |
| `max_len` | 12 | Maximum sequence length |
| `hidden_dim` | 128 | Property head hidden size |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 1e-3 | Adam optimizer learning rate |

## Vocabulary

The vocabulary is automatically built from the sentence config file and includes:

**Special Tokens:**
- `[PAD]` (ID: 0): Padding token
- `[CLS]` (ID: 1): Classification token prepended to all sentences
- `[UNK]` (ID: 2): Unknown token

**Domain Tokens** (from config):
- Colors: red, blue, green
- Sizes: small, medium, large
- Shapes: circle, square, triangle
- Relations: on, above, and, beside

Total vocabulary size: **16 tokens**

## Label Encoding

Each property type has a special `NOT_MENTIONED` class (always class 0) to handle:
- Single-object scenes (object 2 properties are NOT_MENTIONED)
- Missing properties (e.g., color="none" → NOT_MENTIONED)
- No relationship (relationship="none" → NOT_MENTIONED)

Example label spaces:
- **Colors**: {NOT_MENTIONED, red, blue, green} → 4 classes
- **Sizes**: {NOT_MENTIONED, small, medium, large} → 4 classes
- **Shapes**: {NOT_MENTIONED, circle, square, triangle} → 4 classes
- **Relations**: {NOT_MENTIONED, on, above, and, beside} → 5 classes

## Data Format

### Input Scenes (JSON)
```json
{
  "scenes": [
    {
      "objects": [
        {"shape": "circle", "color": "blue", "size": "large"}
      ],
      "relationship": "none"
    },
    {
      "objects": [
        {"shape": "circle", "color": "red", "size": "large"},
        {"shape": "square", "color": "none", "size": "small"}
      ],
      "relationship": "on"
    }
  ]
}
```

### Generated Sentences
- Single object: "large blue circle"
- Two objects: "large red circle on small square"
- Property order: **SIZE, COLOR, SHAPE** (size always before color when both present)

## Model Checkpoint Format

Checkpoints (`.pt` files) contain:
```python
{
    'epoch': int,                    # Training epoch
    'model_state_dict': OrderedDict, # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'val_loss': float,              # Validation loss
    'config': dict,                 # Sentence generation config
}
```

## Next Steps

As noted in the specification, this encoder can be extended to:
1. **2D Image-Like Latents**: Modify encoder to produce spatial feature maps instead of single vectors
2. **Seq-to-Seq Models**: Use encoder as text side of a text-to-image model with an image bottleneck
3. **Multi-Modal Learning**: Pair with visual encoders for vision-language tasks

The current implementation validates that the Transformer encoder can learn meaningful discrete semantic representations from the synthetic language.

## Files Generated During Training

- `checkpoints/vocab.json`: Vocabulary mappings (token2id, id2token)
- `checkpoints/best_model.pt`: Best model checkpoint based on validation loss
- `training_data.json`: Generated scene data for training

## References

- PyTorch Transformer: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
- Scene Generator: See `scene_generator.py` and `cli.py` for data generation
- Specification: See `simple_text_encoder_spec.txt` for full architecture details
