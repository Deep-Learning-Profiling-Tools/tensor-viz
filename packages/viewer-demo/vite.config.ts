import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
    resolve: {
        alias: {
            '@tensor-viz/viewer-core': resolve(__dirname, '../viewer-core/src/index.ts'),
        },
    },
});
