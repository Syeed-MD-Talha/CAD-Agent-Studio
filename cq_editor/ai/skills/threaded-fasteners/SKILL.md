---
name: threaded-fasteners
description: Build bolts, nuts and shafts with real helical threads, matching pitch and useful assembly clearance.
---

Use for explicitly threaded geometry. Keep plain shafts when that is the user's request.

Read `get_examples(name="09_threaded_bolt_and_nut")` for the complete working
Helix/sweep construction. Adapt diameter, pitch, shaft length below the head,
head dimensions, nut dimensions and clearance to the request. State assumptions
when a dimension or standard is unspecified; the example is a simplified thread,
not a certification of a manufacturing standard.

Keep offset profile coordinates using `Polygon(..., align=None)` on `Plane.XZ`.
Sweep with `is_frenet=True`, overlap the ridge root slightly with the core,
then trim the ends. Check that the final bolt is one connected valid solid.
Cut the nut with the matching helix and radial clearance; retain its pitch
when scaling radially. Position parts so their bounds do not intersect unless
the request is for an engaged assembly. Avoid promising print fit without
considering the user's material and process.

Preserve real threads when requested. Repair the failing construction rather
than substituting a smooth cylinder. Validate the complete adapted script
before rendering; show each finished part with a distinct name.
