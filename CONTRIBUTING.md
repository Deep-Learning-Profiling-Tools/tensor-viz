# Contributing

## Setup

Create a Python environment, install the package, and install the frontend
dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
npm install
npx playwright install chromium
git config core.hooksPath .githooks
npm run build
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Run the local demo after setup:

```bash
python demo.py
```

The build step is required in a source checkout because the Python server serves
the frontend from `python/src/tensor_viz/static/`.

## Testing Changes

Run the full test suite before submitting a change:

```bash
npm run test
npm run build
```

`npm run build` is still required when tests pass because it refreshes the demo
assets packaged with the Python server.

Focused checks are useful while iterating:

```bash
npm run typecheck
npm run test --workspace @tensor-viz/viewer-core
npm run test --workspace @tensor-viz/viewer-demo
npm run test:e2e
PYTHONPATH=python/src python -m unittest discover -s python/tests -p 'test_*.py'
```

## Automated Documentation Checks

The mechanical TypeScript documentation checker enforces the JSDoc, comment
density, and helper-function rules from `AGENTS.md`:

```bash
npm run check:ts-docs
npm run check:ts-docs:staged
```

The pre-commit hook runs `npm run check:ts-docs:staged` and
`npm run sync:linear-layout-examples`. Enable it with:

```bash
git config core.hooksPath .githooks
```

The LLM documentation checker is a semantic audit for whether JSDoc text is
useful. It requires `OPENAI_API_KEY` unless you pass `--print-prompt`.

| Goal | Command |
| --- | --- |
| Test the linear-layout parser example | `npm run check:ts-docs:llm-example` |
| Test the example and auto-apply suggested JSDoc replacements | `npm run check:ts-docs:llm-example -- --apply` |
| Audit staged TypeScript files that have no extra unstaged edits | `npm run check:ts-docs:llm` |
| Audit all staged TypeScript blobs from the git index | `npm run check:ts-docs:llm -- --staged` |
| Compare current worktree against a base branch | `npm run check:ts-docs:llm -- --diff --base=origin/main` |
| Audit the full first-party TypeScript codebase | `npm run check:ts-docs:llm -- --all` |

Use `--print-prompt` to inspect prompts without spending tokens:

```bash
npm run check:ts-docs:llm-example -- --print-prompt
npm run check:ts-docs:llm -- --all --limit=4 --print-prompt
```

Use `--apply --stage-applied` when you want accepted replacements staged after
the rewrite:

```bash
npm run check:ts-docs:llm -- --diff --base=origin/main --apply --stage-applied
```

`--apply` only works for worktree-backed audits: default, explicit `--file`,
`--all`, and `--diff`. It intentionally refuses `--staged` because staged mode
reads git-index blobs that may not match the working tree.

Other useful LLM audit options:

```bash
--file=packages/viewer-demo/src/extensions/linear-layout/linear-layout-parser.ts
--symbol=parseLayoutSpecs
--include-direct-helpers
--limit=10
--batch-size=4
--fail-on-error
```

## File Structure

- `packages/viewer-core/src/`: reusable viewer engine, layout math, session
  model, rendering, and core tests.
- `packages/viewer-demo/src/`: browser demo shell, command palette, widget
  lifecycle, extension registry, and app tests.
- `packages/viewer-demo/src/extensions/linear-layout/`: linear-layout extension,
  parser/model code, preset catalog, widgets, and tests.
- `python/src/tensor_viz/`: Python package, session builder, local server, and
  built frontend assets.
- `python/tests/`: Python API and documentation-example tests.
- `docs/`: Sphinx and TypeDoc documentation sources.
- `tools/`: maintenance scripts for checks, generated examples, and demo assets.

## Architecture Guides

Architecture docs live next to the code they describe. Start with:

- [Repository architecture](./ARCHITECTURE.md)
- [Viewer core](./packages/viewer-core/src/ARCHITECTURE.md)
- [Demo app shell](./packages/viewer-demo/src/ARCHITECTURE.md)
- [Linear layout extension](./packages/viewer-demo/src/extensions/linear-layout/ARCHITECTURE.md)
- [Linear layout presets](./packages/viewer-demo/src/extensions/linear-layout/presets/ARCHITECTURE.md)
- [Linear layout widgets](./packages/viewer-demo/src/extensions/linear-layout/widgets/ARCHITECTURE.md)
- [Python package](./python/src/tensor_viz/ARCHITECTURE.md)
- [Maintenance tools](./tools/ARCHITECTURE.md)
- [Browser e2e tests](./packages/viewer-demo/e2e/ARCHITECTURE.md)

When a change modifies a subsystem, update the relevant `ARCHITECTURE.md` in
the same change. Keep user-facing behavior in `docs/` and keep this file focused
on setup, checks, and review expectations.
