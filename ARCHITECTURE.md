The `tensor-viz/` repository is a small monorepo because the project has three jobs that must move together.

First, `packages/viewer-core/` is the reusable viewer engine. It knows how to store tensors, parse tensor-view expressions, compute visible coordinates, and render the result. If a behavior should work in any host application, it belongs in core.

Second, `packages/viewer-demo/` is the browser application. It turns the core viewer into a complete UI with tabs, menus, widgets, a command palette, and optional extensions. The demo can add workflows such as linear-layout presets without teaching core about GPU instruction families.

Third, `python/src/tensor_viz/` is the Python transport layer. It converts Python arrays or metadata into the same manifest format that the TypeScript viewer loads, then serves the built frontend and tensor bytes locally.

The build follows that dependency order. Core builds first, the demo builds against core, and `tools/sync-demo-assets.mjs` copies the built demo into the Python package. That is why `npm run build` is the final required check after code changes: it proves the frontend assets that Python will serve are fresh.

Tests are split by failure mode. Unit tests live beside the TypeScript and Python modules they protect. The Playwright e2e test starts the real demo in Chromium so startup, widget registration, command palette wiring, and canvas paint failures are caught before release.
