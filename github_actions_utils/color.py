"""Color utils for GitHub actions."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
from colorsys import hsv_to_rgb


def score_to_hex_color(score: float, score_min: float, score_max: float) -> str:
    """Convert score to hex color red > bright green."""
    norm_score = max(0, (score - score_min) / (score_max - score_min))
    hsv = (1 / 3 * norm_score, 1, 1)
    rgb = hsv_to_rgb(*hsv)
    rgb_tuple = tuple(int(255 * value) for value in rgb)
    hex_color = f"#{rgb_tuple[0]:02x}{rgb_tuple[1]:02x}{rgb_tuple[2]:02x}"
    return hex_color
