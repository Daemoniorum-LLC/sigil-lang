/**
 * Playwright Configuration for Sigil Playground E2E Tests
 * TDD Phase 4: Playground Integration
 */

const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
    testDir: './',
    fullyParallel: true,
    forbidOnly: !!process.env.CI,
    retries: process.env.CI ? 2 : 0,
    workers: process.env.CI ? 1 : undefined,
    reporter: 'html',

    use: {
        // Base URL for the playground
        baseURL: 'http://localhost:8080',
        trace: 'on-first-retry',
    },

    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] },
        },
        {
            name: 'firefox',
            use: { ...devices['Desktop Firefox'] },
        },
        {
            name: 'webkit',
            use: { ...devices['Desktop Safari'] },
        },
    ],

    // Run a local server before tests
    webServer: {
        command: 'npx serve ../../sigil-lang/website-qliphoth -l 8080',
        url: 'http://localhost:8080',
        reuseExistingServer: !process.env.CI,
        timeout: 120000,
    },
});
