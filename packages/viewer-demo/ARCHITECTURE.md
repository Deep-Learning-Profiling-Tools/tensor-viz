The `packages/viewer-demo/` package is the runnable browser app for tensor-viz.

The package root decides how the app is served and tested. `index.html` is the Vite entry page, `vite.config.ts` resolves the local workspace copy of viewer-core, `src/` contains the application code, `public/` contains static assets copied unchanged by Vite, and `e2e/` contains browser-level tests for the built experience.

Most contributors should start in `src/ARCHITECTURE.md` because UI and extension behavior lives there. Package-root changes should be limited to app startup HTML, Vite behavior, test wiring, or dependencies.
