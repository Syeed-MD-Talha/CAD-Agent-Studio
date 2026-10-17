# Example 3: Extruded Profile
#
# Rather than starting with a primitive shape, you can draw a 2D profile and
# extrude it into a 3D solid. Sketch the cross-section, then give it thickness.
#
# The profile is built as a closed Polyline on a Plane:
#   Plane.XY * Polyline((x, y), ..., close=True)  - trace the outline
#   make_face(...)                                 - fill it into a Face
#
# extrude(face, amount=depth) extrudes the closed profile along Z.
#
# This example draws an L-shaped bracket profile and extrudes it.
#
# For more examples and the full API reference, see
# https://build123d.readthedocs.io/en/latest/

from build123d import *

# Dimensions (mm)
flange_width = 30
flange_height = 5
web_height = 25
web_thickness = 5

# Trace the L-shape profile clockwise starting at the origin.
profile = Polyline(
    (0, 0),
    (flange_width, 0),  # bottom edge of flange
    (flange_width, flange_height),  # right edge of flange
    (web_thickness, flange_height),  # step up to the web
    (web_thickness, flange_height + web_height),  # top of web
    (0, flange_height + web_height),  # top of web (left)
    close=True,
)

result = extrude(make_face(Plane.XY * profile), amount=40)

show_object(result)
