The `viewer-core/src/` directory is the reusable viewer engine. The demo app imports it, but core must not depend on the demo, linear-layout presets, or Python serving details.

The core model starts with session data. `session.ts` normalizes caller-provided bundle and tab specs into the manifest shape loaded by the viewer. `types.ts` defines the shared data contracts that the demo, embedded viewer, and tests all consume.

View parsing and coordinate math live in `view.ts` and `layout.ts`. `view.ts` owns tensor-view expressions, hidden dimensions, grouped axes, and layout-to-tensor coordinate conversion. `layout.ts` owns display positioning, hit testing, axis-family mapping, and index unraveling. Changes to slicing, grouping, axis labels, or visible coordinates should normally be tested in `view.test.ts` or `layout.test.ts`.

Rendering is split between `viewer.ts`, `viewer-graphics.ts`, and `viewer-mesh.ts`. `viewer.ts` owns public viewer state and user-facing methods. The graphics and mesh files turn that state into Three.js objects. Keep rendering code downstream of the model: if a behavior can be expressed as session, view, or layout data, model it there first.

When changing core behavior, add tests next to the affected module. Use `session.test.ts` for manifest normalization, `view.test.ts` for view grammar and slicing behavior, and `layout.test.ts` for coordinate/display math. Then run `npm run test --workspace @tensor-viz/viewer-core` and the repository build.
