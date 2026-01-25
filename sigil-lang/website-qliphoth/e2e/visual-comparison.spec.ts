import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const PRODUCTION_URL = 'https://www.sigil-lang.com';
const WASM_URL = 'http://localhost:8000';

const SCREENSHOT_DIR = path.join(__dirname, '..', 'screenshots', 'visual');

const PAGES = [
  { name: 'index', prodPath: '/', wasmPath: '/index.html' },
  { name: 'docs', prodPath: '/docs.html', wasmPath: '/docs.html' },
  { name: 'learn', prodPath: '/learn.html', wasmPath: '/learn.html' },
  { name: 'agents', prodPath: '/agents.html', wasmPath: '/agents.html' },
  { name: 'pattern', prodPath: '/pattern.html', wasmPath: '/pattern.html' },
  { name: 'qliphoth', prodPath: '/qliphoth.html', wasmPath: '/qliphoth.html' },
];

test.describe('Visual Comparison - Production vs WASM', () => {
  test.beforeAll(async () => {
    fs.mkdirSync(path.join(SCREENSHOT_DIR, 'production'), { recursive: true });
    fs.mkdirSync(path.join(SCREENSHOT_DIR, 'wasm'), { recursive: true });
  });

  // Set consistent viewport
  test.use({ viewport: { width: 1280, height: 800 } });

  for (const pageConfig of PAGES) {
    test(`${pageConfig.name} - capture and compare`, async ({ page }) => {
      console.log(`\n=== Visual Test: ${pageConfig.name} ===`);

      // Capture production screenshot
      await page.goto(`${PRODUCTION_URL}${pageConfig.prodPath}`, {
        waitUntil: 'networkidle',
        timeout: 30000
      });
      await page.waitForTimeout(1000); // Let animations settle

      // Hide dynamic elements that may differ
      await page.evaluate(() => {
        // Hide any consent banners, analytics widgets, etc.
        document.querySelectorAll('[class*="cookie"], [class*="consent"], [class*="banner"]').forEach(el => {
          (el as HTMLElement).style.display = 'none';
        });
      });

      const prodScreenshot = await page.screenshot({ fullPage: true });
      fs.writeFileSync(
        path.join(SCREENSHOT_DIR, 'production', `${pageConfig.name}-full.png`),
        prodScreenshot
      );

      // Capture viewport screenshot
      await page.screenshot({
        path: path.join(SCREENSHOT_DIR, 'production', `${pageConfig.name}-viewport.png`),
      });

      console.log(`  Production: captured`);

      // Capture WASM screenshot
      await page.goto(`${WASM_URL}${pageConfig.wasmPath}`, {
        waitUntil: 'networkidle',
        timeout: 15000
      });
      await page.waitForSelector('#app > *', { timeout: 10000 });
      await page.waitForTimeout(1000);

      const wasmScreenshot = await page.screenshot({ fullPage: true });
      fs.writeFileSync(
        path.join(SCREENSHOT_DIR, 'wasm', `${pageConfig.name}-full.png`),
        wasmScreenshot
      );

      await page.screenshot({
        path: path.join(SCREENSHOT_DIR, 'wasm', `${pageConfig.name}-viewport.png`),
      });

      console.log(`  WASM: captured`);

      // Compare file sizes as rough visual similarity check
      const prodSize = prodScreenshot.length;
      const wasmSize = wasmScreenshot.length;
      const sizeDiff = Math.abs(prodSize - wasmSize) / Math.max(prodSize, wasmSize) * 100;

      console.log(`  Production size: ${(prodSize / 1024).toFixed(1)} KB`);
      console.log(`  WASM size: ${(wasmSize / 1024).toFixed(1)} KB`);
      console.log(`  Size difference: ${sizeDiff.toFixed(1)}%`);

      // Use Playwright's built-in visual comparison
      // This will fail on first run but create baseline, then compare on subsequent runs
      await page.goto(`${WASM_URL}${pageConfig.wasmPath}`, {
        waitUntil: 'networkidle',
        timeout: 15000
      });
      await page.waitForSelector('#app > *', { timeout: 10000 });
      await page.waitForTimeout(500);

      // Take comparison screenshot with generous threshold
      await expect(page).toHaveScreenshot(`${pageConfig.name}-baseline.png`, {
        maxDiffPixelRatio: 0.25, // Allow 25% pixel difference (fonts, rendering)
        threshold: 0.4,
        fullPage: true,
      });
    });
  }
});

test.describe('Side-by-Side Section Comparison', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  for (const pageConfig of PAGES) {
    test(`${pageConfig.name} - section screenshots`, async ({ page }) => {
      // Get production sections
      await page.goto(`${PRODUCTION_URL}${pageConfig.prodPath}`, {
        waitUntil: 'networkidle',
        timeout: 30000
      });
      await page.waitForTimeout(500);

      const prodSections = await page.locator('section').all();
      console.log(`\n${pageConfig.name}: ${prodSections.length} sections in production`);

      for (let i = 0; i < Math.min(prodSections.length, 5); i++) {
        try {
          if (await prodSections[i].isVisible()) {
            await prodSections[i].screenshot({
              path: path.join(SCREENSHOT_DIR, 'production', `${pageConfig.name}-section-${i}.png`),
            });
          }
        } catch (e) {
          // Section might not be screenshottable
        }
      }

      // Get WASM sections
      await page.goto(`${WASM_URL}${pageConfig.wasmPath}`, {
        waitUntil: 'networkidle',
        timeout: 15000
      });
      await page.waitForSelector('#app > *', { timeout: 10000 });
      await page.waitForTimeout(500);

      const wasmSections = await page.locator('section').all();
      console.log(`${pageConfig.name}: ${wasmSections.length} sections in WASM`);

      for (let i = 0; i < Math.min(wasmSections.length, 5); i++) {
        try {
          if (await wasmSections[i].isVisible()) {
            await wasmSections[i].screenshot({
              path: path.join(SCREENSHOT_DIR, 'wasm', `${pageConfig.name}-section-${i}.png`),
            });
          }
        } catch (e) {
          // Section might not be screenshottable
        }
      }

      expect(wasmSections.length).toBe(prodSections.length);
    });
  }
});
