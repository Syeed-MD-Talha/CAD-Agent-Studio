# Example 1: Hello Box
#
# The simplest possible build123d model: a rectangular box.
#
# In Algebra mode every operation returns a new object - there is no
# modelling history to manage. `Box(length, width, height)` creates a
# solid centered on the origin.
#
# For more examples and the full API reference, see
# https://build123d.readthedocs.io/en/latest/

from build123d import *

result = Box(10, 10, 5)

show_object(result)
