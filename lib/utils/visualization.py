"""Validated display options for the bimanual manipulation environment."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping, Optional, Tuple


Color = Optional[Tuple[float, float, float]]


@dataclass(frozen=True)
class VisualizationConfig:
    right_hand_color: Color = None
    left_hand_color: Color = None
    object_color: Color = None
    tool_color: Color = None
    show_reference_skeleton: bool = True
    show_contact_highlight: bool = True
    preserve_textures: bool = False

    @classmethod
    def from_mapping(cls, values=None):
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise ValueError("task.env.visualization must be a mapping")

        fields = {
            "rightHandColor": "right_hand_color",
            "leftHandColor": "left_hand_color",
            "objectColor": "object_color",
            "toolColor": "tool_color",
            "showReferenceSkeleton": "show_reference_skeleton",
            "showContactHighlight": "show_contact_highlight",
            "preserveTextures": "preserve_textures",
        }
        unknown = set(values) - set(fields)
        if unknown:
            raise ValueError("Unknown visualization options: " + ", ".join(sorted(unknown)))

        options = {}
        for name, field in fields.items():
            if name not in values:
                continue
            value = values[name]
            if name.endswith("Color"):
                if value is not None:
                    if not isinstance(value, (list, tuple)) or len(value) != 3 or any(
                        isinstance(channel, bool) or not isinstance(channel, Real)
                        or not math.isfinite(channel) or not 0.0 <= channel <= 1.0
                        for channel in value
                    ):
                        raise ValueError("visualization.{} must be null or three finite RGB values in [0, 1]".format(name))
                    value = tuple(float(channel) for channel in value)
            elif not isinstance(value, bool):
                raise ValueError("visualization.{} must be a boolean".format(name))
            options[field] = value
        return cls(**options)

    def hand_color(self, side):
        return self.right_hand_color if side == "rh" else self.left_hand_color
