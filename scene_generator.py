"""Physical scene data generator.

This module generates synthetic data for physical scenes containing
one or two objects with optional properties (color, size) and spatial
relationships.
"""

import json
import random
from typing import Dict, List, Any, Optional


class SceneObject:
    """Represents an object in a physical scene."""

    def __init__(self, shape: str, color: str = "none", size: str = "none"):
        self.shape = shape
        self.color = color
        self.size = size

    def to_dict(self) -> Dict[str, str]:
        """Convert object to dictionary representation."""
        return {
            "shape": self.shape,
            "color": self.color,
            "size": self.size
        }

    def to_phrase(self) -> str:
        """Convert object to natural language phrase.

        Returns phrase with non-none properties in order: size, color, shape.
        """
        parts = []
        if self.size != "none":
            parts.append(self.size)
        if self.color != "none":
            parts.append(self.color)
        parts.append(self.shape)
        return " ".join(parts)


class Scene:
    """Represents a physical scene with 1-2 objects and optional relationship."""

    def __init__(self, objects: List[SceneObject], relationship: str = "none"):
        self.objects = objects
        self.relationship = relationship

    def to_dict(self) -> Dict[str, Any]:
        """Convert scene to dictionary representation."""
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "relationship": self.relationship
        }

    def to_sentence(self) -> str:
        """Convert scene to natural language sentence.

        Returns sentence describing the scene based on non-none values.
        """
        if len(self.objects) == 1:
            return self.objects[0].to_phrase()

        # Two objects case
        parts = [self.objects[0].to_phrase()]
        if self.relationship != "none":
            parts.append(self.relationship)
        parts.append(self.objects[1].to_phrase())
        return " ".join(parts)


def generate_scenes(config: Dict[str, Any]) -> List[Scene]:
    """Generate scenes based on configuration.

    Args:
        config: Dictionary containing:
            - n: number of scenes to generate
            - p_one: probability of single object scene
            - p_color: probability an object has color
            - p_size: probability an object has size
            - colors: list of possible colors
            - sizes: list of possible sizes
            - shapes: list of possible shapes
            - rels: list of possible spatial relationships

    Returns:
        List of Scene objects.
    """
    scenes = []

    for _ in range(config["n"]):
        # Determine if scene has one or two objects
        is_single_object = random.random() < config["p_one"]

        if is_single_object:
            obj = _generate_object(config)
            scene = Scene(objects=[obj], relationship="none")
        else:
            obj1 = _generate_object(config)
            obj2 = _generate_object(config)
            rel = random.choice(config["rels"])
            scene = Scene(objects=[obj1, obj2], relationship=rel)

        scenes.append(scene)

    return scenes


def _generate_object(config: Dict[str, Any]) -> SceneObject:
    """Generate a single object with random properties.

    Args:
        config: Configuration dictionary with probabilities and value lists.

    Returns:
        SceneObject with randomly assigned properties.
    """
    shape = random.choice(config["shapes"])

    # Determine color based on probability
    if random.random() < config["p_color"]:
        color = random.choice(config["colors"])
    else:
        color = "none"

    # Determine size based on probability
    if random.random() < config["p_size"]:
        size = random.choice(config["sizes"])
    else:
        size = "none"

    return SceneObject(shape=shape, color=color, size=size)


def scenes_to_json(scenes: List[Scene]) -> str:
    """Convert list of scenes to JSON string.

    Args:
        scenes: List of Scene objects.

    Returns:
        JSON string representation of scenes.
    """
    data = {
        "scenes": [scene.to_dict() for scene in scenes]
    }
    return json.dumps(data, indent=2)


def scene_to_sentence(scene_dict: Dict[str, Any]) -> str:
    """Utility function to convert a scene dictionary to a sentence.

    Args:
        scene_dict: Dictionary representation of a scene.

    Returns:
        Natural language sentence describing the scene.
    """
    objects = [SceneObject(**obj) for obj in scene_dict["objects"]]
    scene = Scene(objects=objects, relationship=scene_dict["relationship"])
    return scene.to_sentence()
