import { test } from '@playwright/test';

test('screenshot playground page', async ({ page }) => {
  // Screenshot the restyled playground
  await page.goto('http://localhost:5183/playground/index.html');
  await page.waitForSelector('.panel', { timeout: 10000 });
  await page.waitForTimeout(500);
  await page.screenshot({
    path: 'screenshots/playground-restyled.png',
    fullPage: true
  });
  console.log('Captured: screenshots/playground-restyled.png');
});
