# Example 5: Loft
#
# A loft connects two or more cross-section profiles at different heights
# with a smoothly blended solid. It is useful for organic shapes, aerodynamic
# forms, and transitions between different cross-sections.
#
# The workflow is:
#   1. Sketch a profile on one plane.
#   2. Sketch a second profile on an offset plane.
#   3. Call loft([profile1, profile2]) to blend between them.
#
# This example transitions from a square base to a circular top, like a
# funnel or adapter fitting.
#
# For more examples and the full API reference, see
# https://build123d.readthedocs.io/en/latest/

from build123d import *

base_size = 30  # side length of the square base (mm)
top_radius = 10  # radius of the circular top (mm)
height = 40  # overall height of the loft (mm)

result = loft(
    [
        # First profile: a square on the XY plane at Z=0.
        Plane.XY * Rectangle(base_size, base_size),
        # Second profile: a circle on a plane offset to Z=height.
        Plane.XY.offset(height) * Circle(top_radius),
    ],
    # ruled=False (default) gives a smooth blend.
    # ruled=True would give straight ruled-surface sides instead.
    ruled=False,
)

show_object(result)
