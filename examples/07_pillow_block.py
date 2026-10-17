# Example 7: Pillow Block Bearing Housing
#
# A pillow block is a standard mechanical component: a rectangular base with
# a cylindrical bore that holds a bearing, and mounting holes at the corners.
# It is a classic parametric example because it exercises most of the core
# concepts in a single, practical part:
#
#   - Parametric dimensions
#   - Box primitive
#   - Face and edge selectors
#   - Through-hole (bore)
#   - Counterbored mounting holes on a grid
#   - Fillets
#
# All dimensions are in millimetres.
#
# For more examples and the full API reference, see
# https://build123d.readthedocs.io/en/latest/

from build123d import *

# --- Parameters ---
length = 80.0  # overall length of the base
width = 60.0  # overall width of the base
thickness = 10.0  # thickness of the base plate
bore_dia = 22.0  # diameter of the bearing bore
bolt_dia = 4.4  # mounting bolt clearance hole diameter
cbore_dia = 8.0  # counterbore diameter
cbore_depth = 4.5  # counterbore depth
corner_fillet = 2.0  # radius for the vertical corner fillets
edge_margin = 8.0  # distance from bolt centres to the edge

# Start with a solid rectangular base.
result = Box(length, width, thickness)

# -- Bearing bore -------------------------------------------------------
# Drill a through-hole on the centre axis from the top face.
top = result.faces().filter_by(Axis.Z)[-1]
result -= Plane(top) * Hole(bore_dia / 2, thickness)

# -- Mounting holes ------------------------------------------------------
# Counterbored holes on a grid inset from the edges.
result -= Plane(top) * GridLocations(
    length - edge_margin * 2, width - edge_margin * 2, 2, 2
) * CounterBoreHole(bolt_dia / 2, cbore_dia / 2, cbore_depth, thickness)

# -- Corner fillets ------------------------------------------------------
# Round the four vertical edges (parallel to Z) for a more finished
# appearance and to remove stress concentrations.
result = fillet(result.edges().filter_by(Axis.Z), corner_fillet)

show_object(result)
