import { test } from '@playwright/test';

test('screenshot pattern page', async ({ page }) => {
  // Screenshot the WASM pattern page
  await page.goto('http://localhost:5183/pattern.html');
  await page.waitForSelector('.pattern-section', { timeout: 10000 });
  await page.waitForTimeout(500);
  await page.screenshot({
    path: 'screenshots/wasm-pattern.png',
    fullPage: true
  });
  console.log('Captured: screenshots/wasm-pattern.png');
});
