# Example 4: Holes and Counterbores
#
# Holes are one of the most common features in mechanical parts. build123d
# provides dedicated hole objects a machinist would recognise:
#
#   Hole(radius, depth)                                        - straight hole
#   CounterBoreHole(r, bore_r, bore_depth, depth)              - flat recess
#   CounterSinkHole(r, sink_r, sink_angle, depth)              - conical recess
#
# Holes are placed by multiplying a Plane on the target face with a set of
# Locations, then subtracting:
#   part - Plane(face) * GridLocations(l, w, 2, 2) * CounterBoreHole(...)
#
# For more examples and the full API reference, see
# https://build123d.readthedocs.io/en/latest/

from build123d import *

# Plate dimensions
length = 80.0
width = 60.0
thickness = 10.0

# Fastener dimensions (M4 socket-head cap screw)
bolt_r = 4.4 / 2  # clearance hole radius
cbore_r = 8.0 / 2  # counterbore radius
cbore_depth = 4.5  # counterbore depth

result = Box(length, width, thickness)

# Drill a counterbored hole at each corner of a centred grid on the top face.
top = result.faces().filter_by(Axis.Z)[-1]
result -= Plane(top) * GridLocations(length - 16, width - 16, 2, 2) * (
    CounterBoreHole(bolt_r, cbore_r, cbore_depth, thickness)
)

show_object(result)
