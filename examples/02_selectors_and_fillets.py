# Example 2: Selectors and Fillets
#
# build123d selects faces/edges with ShapeList filters instead of selector
# strings. This example rounds the four vertical edges of a box.
#
# Common patterns:
#   part.edges().filter_by(Axis.Z)  - edges parallel to the Z axis
#   part.edges().filter_by(Axis.X)  - edges parallel to the X axis
#   part.faces().filter_by(Axis.Z)   - faces normal to Z (top/bottom)
#   .group_by(Axis.Z)[-1]            - the face(s) furthest in +Z
#
# fillet(radius) rounds the selected edges.
# chamfer(length) cuts a 45-degree bevel instead.
#
# For more examples and the full API reference, see
# https://build123d.readthedocs.io/en/latest/

from build123d import *

result = Box(10, 10, 5)
# Select the four edges running vertically (parallel to Z) and fillet them.
result = fillet(result.edges().filter_by(Axis.Z), 1.5)

show_object(result)
