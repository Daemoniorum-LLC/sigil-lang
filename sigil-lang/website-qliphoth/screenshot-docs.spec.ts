import { test } from '@playwright/test';

test('screenshot docs page', async ({ page }) => {
  await page.goto('http://localhost:5183/docs.html');
  await page.waitForSelector('.docs-section', { timeout: 10000 });
  await page.waitForTimeout(500);
  await page.screenshot({
    path: 'screenshots/wasm-docs.png',
    fullPage: true
  });
  console.log('Captured: screenshots/wasm-docs.png');
});
