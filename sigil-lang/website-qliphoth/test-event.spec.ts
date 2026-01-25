import { test, expect } from '@playwright/test';

test('event handler function reference works', async ({ page }) => {
  // Collect console logs
  const logs: string[] = [];
  page.on('console', msg => {
    logs.push(`${msg.type()}: ${msg.text()}`);
  });

  await page.goto('http://localhost:5183/test-event.html');

  // Wait for WASM to load and button to appear
  await page.waitForSelector('button', { timeout: 10000 });

  // Should see initial console log (1)
  await page.waitForTimeout(500);
  console.log('Logs before click:', logs);
  expect(logs.some(l => l.includes('[sigil] 1'))).toBe(true);

  // Click the button
  await page.click('button');

  // Wait for handler to execute
  await page.waitForTimeout(500);

  console.log('Logs after click:', logs);

  // Should see 999 in console (from handle_click)
  expect(logs.some(l => l.includes('[sigil] 999'))).toBe(true);

  console.log('Event handler test passed!');
});
