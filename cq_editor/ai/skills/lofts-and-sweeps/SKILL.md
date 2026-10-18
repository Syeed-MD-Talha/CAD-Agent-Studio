---
name: lofts-and-sweeps
description: Model changing cross-sections, hollow lofts and curved sweeps with correctly placed profiles and paths.
---

For lofts, read `get_examples(name="05_loft")`. For curved sweeps, read
`get_examples(name="06_sweep")`. Load only the example needed for the request.

Place loft profiles at their actual heights using `Plane.XY.offset(z) * sketch`.
Keep compatible profile topology and distinct section planes. Start with a
small number of sections; add detail only when it changes the requested shape.
For a hollow vessel, subtract an inner loft with a shorter bottom extent so
the floor remains closed. Check positive wall thickness through every section.

For a sweep, place the profile at the path start with an appropriate normal;
a profile left at the origin can produce a misplaced or failed solid. Use the
global `sweep(profile, path=path)`. Frenet framing is useful for helical ridges;
choose framing that fits other paths rather than applying it universally.

Use `plane * sketch` or `shape.located(plane.location)` as appropriate.
`located` takes a Location, not a Plane. If a build fails, inspect the exact
section, path or self-intersection before retrying; preserve the requested form.
