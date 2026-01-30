import { test, expect } from '@playwright/test';

const pages = [
  { name: 'index', url: '/' },
  { name: 'docs', url: '/docs.html' },
  { name: 'learn', url: '/learn.html' },
  { name: 'agents', url: '/agents.html' },
  { name: 'pattern', url: '/pattern.html' },
];

for (const page of pages) {
  test(`audit ${page.name} page for JS errors`, async ({ page: browserPage }) => {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Capture console errors
    browserPage.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(`[ERROR] ${msg.text()}`);
      } else if (msg.type() === 'warning') {
        warnings.push(`[WARN] ${msg.text()}`);
      }
    });

    // Capture page errors (uncaught exceptions)
    browserPage.on('pageerror', err => {
      errors.push(`[PAGE ERROR] ${err.message}`);
    });

    // Navigate to page
    await browserPage.goto(page.url, { waitUntil: 'networkidle' });

    // Wait a bit for WASM to initialize
    await browserPage.waitForTimeout(2000);

    // Log all errors found
    if (errors.length > 0) {
      console.log(`\n=== ${page.name.toUpperCase()} PAGE ERRORS ===`);
      errors.forEach(e => console.log(e));
    }

    if (warnings.length > 0) {
      console.log(`\n=== ${page.name.toUpperCase()} PAGE WARNINGS ===`);
      warnings.forEach(w => console.log(w));
    }

    // Fail if there are errors
    expect(errors, `${page.name} page has JavaScript errors`).toHaveLength(0);
  });
}
