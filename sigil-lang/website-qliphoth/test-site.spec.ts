import { test, expect } from '@playwright/test';

test('sigil-lang.com WASM site renders correctly', async ({ page }) => {
  // Navigate to the test page
  await page.goto('http://localhost:5183/test-qliphoth.html');

  // Wait for WASM to load and render
  await page.waitForSelector('#app div', { timeout: 10000 });

  // Take a full page screenshot
  await page.screenshot({
    path: 'screenshots/site-current.png',
    fullPage: true
  });

  // Check status shows success
  const status = await page.locator('#status').textContent();
  console.log('Status:', status);

  // Verify key elements exist
  const docsLink = page.locator('a[href="/pages/docs.html"]');
  const githubLink = page.locator('a[href*="github.com"]');

  expect(await docsLink.count()).toBeGreaterThanOrEqual(1);
  expect(await githubLink.count()).toBeGreaterThanOrEqual(1);

  console.log('Docs link href:', await docsLink.first().getAttribute('href'));
  console.log('GitHub link href:', await githubLink.first().getAttribute('href'));

  // Log what rendered in #app
  const appContent = await page.locator('#app').innerHTML();
  console.log('App HTML length:', appContent.length);
});

test('compare with minimal test', async ({ page }) => {
  await page.goto('http://localhost:5183/test-minimal.html');
  await page.waitForSelector('#app div', { timeout: 10000 });

  await page.screenshot({
    path: 'screenshots/minimal-current.png',
    fullPage: true
  });

  const status = await page.locator('#status').textContent();
  console.log('Minimal status:', status);
});
