The `viewer-demo/src/` directory is the browser app that turns tensor-viz's reusable viewer engine into a mountable demo shell.

The demo now has two layers. `app-entry.ts`, `app-shell.ts`, `app-extension.ts`, and `main.ts` create the page, command palette, sidebar, tab state, viewer instance, and extension lifecycle hooks. Feature-specific behavior lives under `extensions/` and registers widgets, controls, session migration, and render hooks through the extension API.

The stock app currently registers one extension: `extensions/linear-layout/`. That extension owns the linear-layout model, preset catalog, sidebar widgets, hover popup, selection synchronization, and baked fallback tabs. Keep new linear-layout behavior there instead of adding new branches to `app-entry.ts`.

`app-entry.ts` should stay shell-shaped. It may ask extensions whether a widget is visible, let them capture tab state, or forward viewer events, but it should not know instruction families, preset fields, or linear-layout tensor metadata. That keeps future extensions from needing to edit the same central file for every new behavior.

The important local guides are:

- `extensions/linear-layout/ARCHITECTURE.md` explains the linear-layout extension boundary.
- `extensions/linear-layout/presets/ARCHITECTURE.md` explains how preset data and selector fields are added.
- `extensions/linear-layout/widgets/ARCHITECTURE.md` explains how sidebar widgets are split and where UI changes belong.

When changing shell behavior, add the smallest app-level test that protects the extension contract. When changing linear-layout parsing or composition, update `extensions/linear-layout/linear-layout.test.ts`.
