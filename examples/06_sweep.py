# Example 6: Sweep
#
# A sweep moves a 2D cross-section along a 3D path. Unlike a straight
# extrude, the path can curve, making sweep ideal for pipes, channels,
# handles, and wiring conduits.
#
# The workflow:
#   1. Build the path as a curve (here a quarter-circle arc in the XZ plane).
#   2. Draw the profile at the start of the path.
#   3. Call sweep(profile, path) to move the profile along the path.
#
# This example creates a curved pipe - a circular cross-section swept along
# a quarter-circle arc.
#
# For more examples and the full API reference, see
# https://build123d.readthedocs.io/en/latest/

from build123d import *

pipe_radius = 3  # outer radius of the pipe cross-section (mm)
sweep_radius = 30  # radius of the quarter-circle sweep path (mm)

# Build the sweep path: a quarter-circle arc in the XZ plane from
# (sweep_radius, 0) to (0, sweep_radius).
path = Plane.XZ * JernArc(
    (sweep_radius, 0), (0, sweep_radius), radius=sweep_radius, arc_size=90
)

# Circular cross-section positioned at the start of the path.
profile = Plane.XY * Circle(pipe_radius)
profile = profile.move(Location((sweep_radius, 0, 0)))

result = sweep(profile, path)

show_object(result)
