The `viewer-demo/src/` directory is the browser app that turns tensor-viz's reusable viewer engine into the full linear-layout demo.

The demo has two layers. `app-entry.ts`, `app-shell.ts`, and `main.ts` create the page, command palette, sidebar, tab state, and viewer instance. The linear-layout files then teach that app how to parse layout specs, build intermediate tensors, and synchronize the viewer with the sidebar.

`linear-layout.ts` is the center of the linear-layout model. It owns the compose-layout parser, operation evaluator, preset normalization, matrix previews, generated Python, and metadata that gets embedded into viewer tabs. If a change affects layout syntax, composition semantics, output labels, matrix blocks, or the session metadata emitted for a rendered layout, start there.

`linear-layout-state.ts` is the bridge from saved viewer tabs and browser storage back into live sidebar state. It should stay focused on cloning, validation, tab synchronization, and persistence. `linear-layout-viewer-sync.ts` is the bridge in the other direction: it translates current runtime metadata into viewer labels, colors, selection, hover popups, and multi-input display state.

The subdirectories keep specialized workflows out of the central model:

- `linear-layout-presets/ARCHITECTURE.md` explains how preset data and selector fields are added.
- `widgets/ARCHITECTURE.md` explains how sidebar widgets are split and where UI changes belong.

When changing linear-layout parsing or composition, update `linear-layout.test.ts` with the smallest test that captures the behavior. Good tests here usually build a `ComposeRuntime`, inspect emitted metadata or generated Python, and then assert the mapping behavior that would break in the UI.
