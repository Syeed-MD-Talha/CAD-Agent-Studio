# Example 8: Assembly
#
# Most designs consist of more than one part. In build123d an assembly is
# simply a Compound that groups parts as children - every child keeps a
# location relative to its parent.
#
#   assembly = Compound(label="...", children=[part1, part2])
#
# CQ-editor explodes assemblies into one tree item per part, so call
# show_object once per part (or once for the whole assembly).
#
# This example builds a simple shaft + housing assembly.
#
# For more examples and the full API reference, see
# https://build123d.readthedocs.io/en/latest/

from build123d import *

# Housing: a block with a bore the shaft passes through.
housing = Box(40, 30, 20)
top = housing.faces().filter_by(Axis.Z)[-1]
housing -= Plane(top) * Hole(6, 20)
housing.label = "housing"
housing.color = Color("steelblue")

# Shaft: a cylinder standing proud of the housing.
shaft = Cylinder(5, 34, align=(Align.CENTER, Align.CENTER, Align.MIN))
shaft = shaft.move(Location((0, 0, -7)))
shaft.label = "shaft"
shaft.color = Color("orange")

assembly = Compound(label="shaft_assembly", children=[housing, shaft])

show_object(housing, name="housing")
show_object(shaft, name="shaft")
result = assembly
