import { test } from '@playwright/test';

test('screenshot learn page', async ({ page }) => {
  // Screenshot the WASM learn page
  await page.goto('http://localhost:5183/learn.html');
  await page.waitForSelector('.chapter', { timeout: 10000 });
  await page.waitForTimeout(500);
  await page.screenshot({
    path: 'screenshots/wasm-learn.png',
    fullPage: true
  });
  console.log('Captured: screenshots/wasm-learn.png');
});
