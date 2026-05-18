The `viewer-demo/src/` directory is the browser app that turns tensor-viz's reusable viewer engine into a mountable demo shell.

The demo now has two layers. `app-entry.ts`, `app-shell.ts`, `app-extension.ts`, `registered-extensions.ts`, and `main.ts` create the page, command palette, sidebar, tab state, viewer instance, and extension lifecycle hooks. Feature-specific behavior should live in downstream packages and register widgets, controls, session migration, tensor-view contributions, inspector rows, and render hooks through the extension API.

The stock app currently registers no domain-specific extensions. LL-viz imports `startDemoApp(...)` from `@tensor-viz/viewer-demo` and passes its linear-layout extension factory at startup, which keeps GPU instruction families and preset catalogs out of tensor-viz.

`registered-extensions.ts` is the app's extension registry. New demo features should contribute a factory there with widget slots and a `create(...)` function. `app-entry.ts` should stay shell-shaped: it may ask extensions whether a widget is visible, let them capture tab state, collect extra tensor-view sliders, collect inspector coordinate rows, or forward viewer events, but it should not know instruction families, preset fields, or linear-layout tensor metadata. That keeps future extensions from needing to edit the same central file for every new behavior.

When changing shell behavior, add the smallest app-level test that protects the extension contract. When changing a downstream workflow, update that workflow's own package and tests instead of adding branches here.
