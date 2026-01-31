import { test } from '@playwright/test';

test('screenshot playground with content', async ({ page }) => {
  await page.goto('http://localhost:5183/playground/index.html');

  // Wait for editor to be ready
  await page.waitForSelector('.cm-editor', { timeout: 15000 });
  await page.waitForTimeout(1500);

  await page.screenshot({
    path: 'screenshots/playground-full.png',
    fullPage: true
  });
  console.log('Captured: screenshots/playground-full.png');
});
