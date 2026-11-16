"""Direction-independent stroke interpolation used by the native Qt sketch pad."""
import numpy as np


def interpolate_stroke(values, previous_index, previous_value, index, value):
    """Fill samples between two mouse events, preserving their actual endpoints."""
    if previous_index is None:
        values[index] = value
    else:
        a, b = sorted((int(previous_index), int(index)))
        left, right = ((previous_value, value) if previous_index <= index
                       else (value, previous_value))
        values[a:b + 1] = np.linspace(left, right, b - a + 1)
    return values
