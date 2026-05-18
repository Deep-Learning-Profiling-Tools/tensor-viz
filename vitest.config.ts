import { defineConfig } from 'vitest/config';

// keep workspace test runs anchored inside tensor-viz even when this checkout is
// nested below another Vite app such as LL-viz.
export default defineConfig({});
