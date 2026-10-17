import pytest

from build123d import Plane, Rectangle

from cq_editor.cq_utils import to_compound


def test_to_compound_applies_sketch_placement():
    sk = Plane.XZ * Rectangle(1, 1)
    f = to_compound(sk).faces()[0]

    assert f.area == pytest.approx(1)
    assert tuple(f.normal_at()) == pytest.approx((0, -1, 0))
