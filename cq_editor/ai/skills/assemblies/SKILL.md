---
name: assemblies
description: Create multiple distinct CAD parts with intentional placement, meaningful names and assembly relationships.
---

Read `get_examples(name="08_assembly")` for a complete multipart example.

Construct parts in convenient local coordinates, then position them with
`Pos` and `Rot`. Account for centered primitives: explicitly choose Align.MIN
when the specified height should begin at z=0. Determine whether dimensions
refer to individual parts or the complete assembly.

Use bounding extents and mating surfaces to place parts. Side-by-side views
need a visible gap; engaged assemblies need matching axes, pitch where
threaded, and the requested fit or clearance. Avoid arbitrary translations
that hide collisions or misrepresent the mechanical relationship.

Keep independent parts separate instead of fusing touching components into one
solid. Use `Compound(children=[part1, part2])` for a combined result and
`show_object(part, name="...")` for separately selectable components.
Check intended solid count, dimensions and placement before declaring success.
