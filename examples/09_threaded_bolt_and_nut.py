# M12 x 1.75 bolt and matching nut, with real helical geometry.
# Dimensions are millimeters. These simplified thread flanks omit the
# manufacturing root radii and end chamfers; allow clearance for printing.
from build123d import *
import math

diameter = 12.0
pitch = 1.75
shaft_length = 45.0  # measured below the head
head_flats, head_height = 18.0, 7.5
nut_flats, nut_height = 18.0, 10.0
clearance = 0.2  # radial clearance in the nut
radius = diameter / 2
root_radius = radius - 0.61343 * pitch


def threaded_shaft(length):
    # Polygon coordinates must retain their radial offset (align=None).
    # An XZ section at the helix start follows the Z axis with Frenet framing.
    profile = Plane.XZ * Polygon(
        (root_radius - 0.05, -pitch * 0.40),
        (radius, -pitch / 16),
        (radius, pitch / 16),
        (root_radius - 0.05, pitch * 0.40),
        align=None,
    )
    path = Helix(pitch=pitch, height=length, radius=root_radius)
    ridge = sweep(profile, path=path, is_frenet=True)
    core = Cylinder(root_radius, length, align=(Align.CENTER, Align.CENTER, Align.MIN))
    # Trim the partial turns at both ends to the requested shaft length.
    envelope = Cylinder(radius + 0.1, length, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return (core + ridge) & envelope


shaft = threaded_shaft(shaft_length)
head = Pos(0, 0, -head_height) * extrude(
    RegularPolygon(head_flats / math.sqrt(3), side_count=6), amount=head_height
)
bolt = head + shaft

nut_blank = extrude(RegularPolygon(nut_flats / math.sqrt(3), side_count=6), amount=nut_height)
# Use a longer copy of the same screw as a cutter. Enlarging it radially
# provides clearance while retaining exactly the same helical pitch.
cutter = Pos(0, 0, -pitch) * threaded_shaft(nut_height + 2 * pitch)
cutter = scale(cutter, by=(1 + clearance / radius, 1 + clearance / radius, 1))
nut = Pos(30, 0, 0) * (nut_blank - cutter)

result = Compound(children=[bolt, nut])
show_object(bolt, name="M12 bolt")
show_object(nut, name="M12 nut")
