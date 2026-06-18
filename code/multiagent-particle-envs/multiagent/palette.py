"""Color palette constants and helpers for the multi-agent rendering.

This module stores color constants in RGB format and provides helpers to
dynamically compute color adjustments (e.g., creating lighter shades).
"""

BACKGROUND = (25 / 255.0, 28 / 255.0, 33 / 255.0, 1.0)  # #191c21
TARGET_BASE = (221 / 255.0, 54 / 255.0, 159 / 255.0, 1.0)  # #dd369f
TARGET_ACTIVATED = (165 / 255.0, 148 / 255.0, 90 / 255.0, 1.0)  # #a5945a
AGENT_BASE = (86 / 255.0, 163 / 255.0, 254 / 255.0, 1.0)  # #56a3fe
LEADER_BASE = (120 / 255.0, 120 / 255.0, 253 / 255.0, 1.0)  # #5b94fd

AGENT_LIGHTER_FACTOR = 0.1


def get_lighter_color(color, factor):
    """Computes a lighter shade of a given RGB color.

    Interpolates between the given color and white (1.0, 1.0, 1.0)
    using the specified factor.

    Args:
        color (tuple or list): Base color in RGB (values between 0.0 and 1.0).
        factor (float): Interpolation factor (0.0 returns base color, 1.0 returns white).

    Returns:
        list: The adjusted RGB color values.
    """
    adjusted = [c + (1.0 - c) * factor for c in color[:3]]
    if len(color) == 4:
        return adjusted + [color[3]]
    return adjusted
