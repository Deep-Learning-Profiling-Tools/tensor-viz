The `packages/viewer-core/` package is the TypeScript library boundary for the viewer engine.

The package root is intentionally thin. `package.json` describes how the library builds, tests, and exposes source during local workspace development. `tsconfig.json` points TypeScript at `src/`, where the real architecture is documented in `src/ARCHITECTURE.md`.

Keep package-root changes about packaging, dependency boundaries, or build behavior. Viewer behavior should live under `src/` so tests, API exports, and architecture notes stay next to the implementation.
