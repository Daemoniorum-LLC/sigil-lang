import { test } from '@playwright/test';

test('screenshot both sites for comparison', async ({ page }) => {
  // Screenshot the flat HTML site
  await page.goto('http://localhost:5184/');
  await page.waitForTimeout(1000);
  await page.screenshot({
    path: 'screenshots/html-site-full.png',
    fullPage: true
  });
  console.log('Captured: screenshots/html-site-full.png');

  // Screenshot the WASM site
  await page.goto('http://localhost:5183/test-qliphoth.html');
  await page.waitForSelector('#app div', { timeout: 10000 });
  await page.waitForTimeout(500);
  await page.screenshot({
    path: 'screenshots/wasm-site-full.png',
    fullPage: true
  });
  console.log('Captured: screenshots/wasm-site-full.png');
});
