---
name: geometry-repair
description: Repair build123d validation failures, incompatible API calls, invalid geometry and kernel timeouts without losing requested features.
---

Start with the error summary, failing script line and installed build123d
version supplied by validation. The failed script is in the previous reply or
tool arguments; it was not written into the editor. Fetching an empty editor
does not recover that code.

Common API corrections:
- `shape.located(plane.location)` for an absolute plane placement; `plane *
  sketch` or `Pos(...) * shape` for the corresponding transformations.
- Global `extrude(sketch, amount=height)`, not `sketch.extrude(amount=height)`.
- `RegularPolygon(radius, side_count=6)` for a hexagon. `Polygon` takes vertices;
  use `align=None` when those coordinates must keep a radial or other offset.

For boolean failures, inspect overlap, tangent contacts and thin remnants.
For fillet failures, check the selected edge set and reduce an infeasible radius
while explaining any deviation from specified dimensions. For a native crash or
timeout, simplify construction and validate again in isolation before rendering.
Do not repeatedly submit the unchanged failing script.

Return or validate one complete corrected script. Keep the user's features and
dimensions; explain unresolved requirements instead of inventing success.
After a novel repair succeeds, an optional learned skill should capture the
general method and an executable example rather than the full chat transcript.
