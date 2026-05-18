The `.githooks/` directory contains local git hooks that run publish-friendly
checks before a commit is created.

The current pre-commit hook runs `npm run check:ts-docs:staged`. That audits
staged TypeScript files for the documentation and helper-use rules in
`AGENTS.md` without forcing a full codebase audit on every small commit.
