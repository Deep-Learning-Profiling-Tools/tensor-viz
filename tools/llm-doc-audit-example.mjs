import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptPath = resolve(dirname(fileURLToPath(import.meta.url)), 'check-ts-docs-llm.mjs');
const result = spawnSync(
    process.execPath,
    [
        scriptPath,
        '--file=packages/viewer-demo/src/extensions/linear-layout/linear-layout-parser.ts',
        '--symbol=parseLayoutSpecs',
        '--include-direct-helpers',
        '--batch-size=8',
        ...process.argv.slice(2),
    ],
    { stdio: 'inherit' },
);

process.exit(result.status ?? 1);
