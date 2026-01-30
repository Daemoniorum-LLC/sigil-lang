import { test } from '@playwright/test';

test('screenshot agents pages', async ({ page }) => {
  // Screenshot the WASM agents page
  await page.goto('http://localhost:5183/agents.html');
  await page.waitForSelector('.layer-section', { timeout: 10000 });
  await page.waitForTimeout(500);
  await page.screenshot({
    path: 'screenshots/wasm-agents.png',
    fullPage: true
  });
  console.log('Captured: screenshots/wasm-agents.png');
});
