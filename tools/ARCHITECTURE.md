The `tools/` directory contains repository maintenance scripts, not runtime library code.

`sync-demo-assets.mjs` runs after the demo build. It copies Vite's generated files into `python/src/tensor_viz/static/` so the Python package serves the same frontend that local development tested.

`sync-linear-layout-examples.py` keeps baked linear-layout demo tabs synchronized with the LL-viz Python demo source when that source is available. It rewrites only the marked generated block in the linear-layout extension, which keeps hand-written TypeScript separate from generated examples.

Tooling here should stay deterministic and small. If a script starts needing app state or viewer internals, move that behavior into the package that owns the concept and let the tool call the package-level API.

`check-ts-docs.mjs` is the mechanical guard for the TypeScript commenting rules in `AGENTS.md`. It parses source files with the TypeScript AST, not regular expressions, so declarations are checked as declarations: functions, function-valued variables, classes, interfaces, type aliases, enums, and methods. In normal mode, it audits every first-party TypeScript source file. With explicit file or directory arguments, it audits only those paths. In `--staged` mode, it audits every staged TypeScript source file as a whole file, which means touching a file also makes that file's existing undocumented helpers visible.

The same checker enforces two non-docstring rules. First, source files must meet a configurable comment-line ratio, currently 10 percent. Second, tiny non-exported top-level helper functions must have at least three local references. A tiny helper with fewer call sites is usually a sectioning helper, not a real abstraction; interface-boundary exceptions can be marked with a JSDoc `@interfaceBoundary` tag. Larger algorithmic routines are still checked through JSDoc and comment density instead of being forced inline.

The full `npm run check:ts-docs` command is intentionally broad and is expected to fail until the legacy TypeScript files are cleaned up. The staged command is the publish-friendly gate: it prevents commits from changing a TypeScript source file while leaving that same file below the documented standard. For targeted cleanup, run `node tools/check-ts-docs.mjs packages/viewer-demo/src/app-entry.ts` or pass any other file/directory path.
