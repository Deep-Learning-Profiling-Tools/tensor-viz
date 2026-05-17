import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
    testDir: './packages/viewer-demo/e2e',
    reporter: [['list']],
    use: {
        baseURL: 'http://127.0.0.1:5173',
        screenshot: 'only-on-failure',
        trace: 'on-first-retry',
        ...devices['Desktop Chrome'],
    },
    webServer: {
        command: 'npm run dev --workspace @tensor-viz/viewer-demo -- --host 127.0.0.1 --port 5173',
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
        url: 'http://127.0.0.1:5173',
    },
});
