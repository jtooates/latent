# Physical Scene Data Generator

A Python tool for generating synthetic data representing physical scenes with objects and spatial relationships.

## Overview

This data generator creates scenes containing one or two objects. Each object has a shape and can optionally have a color and/or size. When two objects are present, they have a spatial relationship.

## Installation

No external dependencies required - uses only Python standard library.

```bash
# Make CLI executable (optional)
chmod +x cli.py
```

## Usage

### As a CLI Tool

```bash
python cli.py example_config.json
```

Options:
- `-o, --output FILE`: Write output to file instead of stdout
- `--sentences`: Also print natural language sentences for each scene

Examples:

```bash
# Output to stdout
python cli.py example_config.json

# Save to file
python cli.py example_config.json -o scenes.json

# Show sentences
python cli.py example_config.json --sentences
```

### As a Library

```python
import json
from scene_generator import generate_scenes, scenes_to_json, scene_to_sentence

# Load config
with open('example_config.json') as f:
    config = json.load(f)

# Generate scenes
scenes = generate_scenes(config)

# Convert to JSON
json_output = scenes_to_json(scenes)
print(json_output)

# Convert individual scenes to sentences
for scene in scenes:
    print(scene.to_sentence())

# Or convert from dict
scene_dict = {"objects": [{"shape": "circle", "color": "red", "size": "none"}], "relationship": "none"}
sentence = scene_to_sentence(scene_dict)
```

## Configuration

The configuration file is a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `n` | int | Number of scenes to generate |
| `p_one` | float | Probability that a scene has only one object (0-1) |
| `p_color` | float | Probability that an object has a color (0-1) |
| `p_size` | float | Probability that an object has a size (0-1) |
| `colors` | list[str] | List of possible colors |
| `sizes` | list[str] | List of possible sizes |
| `shapes` | list[str] | List of possible shapes |
| `rels` | list[str] | List of possible spatial relationships |

Example configuration:

```json
{
  "n": 10,
  "p_one": 0.3,
  "p_color": 0.7,
  "p_size": 0.5,
  "colors": ["red", "blue", "green"],
  "sizes": ["small", "large"],
  "shapes": ["circle", "square", "triangle"],
  "rels": ["above", "below", "left of", "right of"]
}
```

## Output Format

The generator outputs JSON with the following structure:

```json
{
  "scenes": [
    {
      "objects": [
        {"shape": "circle", "color": "red", "size": "none"}
      ],
      "relationship": "none"
    },
    {
      "objects": [
        {"shape": "square", "color": "blue", "size": "large"},
        {"shape": "triangle", "color": "none", "size": "small"}
      ],
      "relationship": "above"
    }
  ]
}
```

### Sentences

The utility function converts scenes to natural language:

- Single object: `"red circle"` or `"large square"` or `"triangle"`
- Two objects: `"red circle above large square"` or `"circle below triangle"`

Only non-"none" properties are included in the sentence.

## Project Structure

```
.
├── scene_generator.py   # Core library
├── cli.py              # Command-line interface
├── example_config.json # Sample configuration
└── README.md          # This file
```
