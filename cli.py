#!/usr/bin/env python3
"""Command-line interface for the physical scene data generator."""

import argparse
import json
import sys
from pathlib import Path
from scene_generator import generate_scenes, scenes_to_json


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic physical scene data"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--sentences",
        action="store_true",
        help="Also print sentences for each scene"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate required config fields
    required_fields = ["n", "p_one", "p_color", "p_size", "colors", "sizes", "shapes", "rels"]
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        print(f"Error: Missing required config fields: {', '.join(missing_fields)}", file=sys.stderr)
        sys.exit(1)

    # Generate scenes
    scenes = generate_scenes(config)
    json_output = scenes_to_json(scenes)

    # Output results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write(json_output)
        print(f"Generated {len(scenes)} scenes -> {args.output}")
    else:
        print(json_output)

    # Optionally print sentences
    if args.sentences:
        print("\nSentences:", file=sys.stderr)
        for i, scene in enumerate(scenes, 1):
            print(f"{i}. {scene.to_sentence()}", file=sys.stderr)


if __name__ == "__main__":
    main()
