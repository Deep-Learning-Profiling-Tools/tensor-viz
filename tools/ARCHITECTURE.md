The `tools/` directory contains repository maintenance scripts, not runtime library code.

`sync-demo-assets.mjs` runs after the demo build. It copies Vite's generated files into `python/src/tensor_viz/static/` so the Python package serves the same frontend that local development tested.

`sync-linear-layout-examples.py` keeps baked linear-layout demo tabs synchronized with the LL-viz Python demo source when that source is available. It rewrites only the marked generated block in the linear-layout extension, which keeps hand-written TypeScript separate from generated examples.

Tooling here should stay deterministic and small. If a script starts needing app state or viewer internals, move that behavior into the package that owns the concept and let the tool call the package-level API.

`check-ts-docs.mjs` is the mechanical guard for the TypeScript commenting rules in `AGENTS.md`. It parses source files with the TypeScript AST, not regular expressions, so declarations are checked as declarations: functions, function-valued variables, classes, interfaces, type aliases, enums, and methods. In normal mode, it audits every first-party TypeScript source file. In `--staged` mode, it only audits declarations whose declaration line is in the staged diff, which lets the pre-commit hook raise the bar incrementally without requiring every legacy helper to be fixed before a small preset edit can land.

The same checker enforces two non-docstring rules. First, source files must meet a configurable comment-line ratio, currently 10 percent. In staged mode this ratio applies to changed nonblank lines and is skipped for tiny edits, which avoids forcing a comment on one-line metadata tweaks. Second, non-exported top-level helper functions must have at least three local references. A helper with fewer call sites is usually a sectioning helper, not a real abstraction; interface-boundary exceptions can be marked with a JSDoc `@interfaceBoundary` tag.

The full `npm run check:ts-docs` command is intentionally stricter than the pre-commit hook and is expected to fail until the legacy TypeScript files are cleaned up. The staged command is the publish-friendly gate: it keeps new or edited declarations from adding more undocumented functions, low-comment code, or one-off helpers while allowing existing debt to be paid down file by file.
