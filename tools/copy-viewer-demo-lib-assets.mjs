import { copyFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';

const packageRoot = resolve(import.meta.dirname, '..', 'packages', 'viewer-demo');

// compiled app-entry.js imports ./styles.css, so the package lib directory must
// carry the stylesheet beside the emitted JavaScript entrypoints.
await mkdir(resolve(packageRoot, 'lib'), { recursive: true });
await copyFile(resolve(packageRoot, 'src', 'styles.css'), resolve(packageRoot, 'lib', 'styles.css'));
