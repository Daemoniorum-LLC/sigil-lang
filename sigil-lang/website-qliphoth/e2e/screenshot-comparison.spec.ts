import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const PRODUCTION_URL = 'https://www.sigil-lang.com';
const QLIPHOTH_URL = 'http://localhost:8000';

const PAGES = [
  { path: '/', name: 'home' },
  { path: '/learn.html', name: 'learn' },
  { path: '/docs.html', name: 'docs' },
  { path: '/playground/', name: 'playground' },
];

const SCREENSHOT_DIR = path.join(__dirname, '..', 'screenshots');

test.describe('Production Site Screenshots', () => {
  test.beforeAll(async () => {
    // Ensure directories exist
    fs.mkdirSync(path.join(SCREENSHOT_DIR, 'production'), { recursive: true });
  });

  for (const page of PAGES) {
    test(`capture ${page.name}`, async ({ page: browserPage }) => {
      await browserPage.goto(`${PRODUCTION_URL}${page.path}`, { waitUntil: 'networkidle' });
      await browserPage.waitForTimeout(1000); // Let animations settle

      // Full page screenshot
      await browserPage.screenshot({
        path: path.join(SCREENSHOT_DIR, 'production', `${page.name}-full.png`),
        fullPage: true,
      });

      // Viewport screenshot
      await browserPage.screenshot({
        path: path.join(SCREENSHOT_DIR, 'production', `${page.name}-viewport.png`),
      });

      // Get page sections and screenshot each
      const sections = await browserPage.locator('section, header, footer, main, .hero, .section').all();
      for (let i = 0; i < Math.min(sections.length, 10); i++) {
        try {
          const section = sections[i];
          if (await section.isVisible()) {
            await section.screenshot({
              path: path.join(SCREENSHOT_DIR, 'production', `${page.name}-section-${i}.png`),
            });
          }
        } catch (e) {
          // Section might not be screenshottable
        }
      }
    });
  }
});

test.describe('Qliphoth Site Screenshots', () => {
  test.beforeAll(async () => {
    fs.mkdirSync(path.join(SCREENSHOT_DIR, 'qliphoth'), { recursive: true });
  });

  // Map qliphoth paths (different structure than production)
  const QLIPHOTH_PAGES = [
    { path: '/', name: 'home' },
    { path: '/learn.html', name: 'learn' },
    { path: '/docs.html', name: 'docs' },
    { path: '/playground/', name: 'playground' },
  ];

  for (const page of QLIPHOTH_PAGES) {
    test(`capture ${page.name}`, async ({ page: browserPage }) => {
      try {
        await browserPage.goto(`${QLIPHOTH_URL}${page.path}`, {
          waitUntil: 'networkidle',
          timeout: 10000
        });
        await browserPage.waitForTimeout(1000);

        // Full page screenshot
        await browserPage.screenshot({
          path: path.join(SCREENSHOT_DIR, 'qliphoth', `${page.name}-full.png`),
          fullPage: true,
        });

        // Viewport screenshot
        await browserPage.screenshot({
          path: path.join(SCREENSHOT_DIR, 'qliphoth', `${page.name}-viewport.png`),
        });

        // Get page sections
        const sections = await browserPage.locator('section, header, footer, main, .hero, .section').all();
        for (let i = 0; i < Math.min(sections.length, 10); i++) {
          try {
            const section = sections[i];
            if (await section.isVisible()) {
              await section.screenshot({
                path: path.join(SCREENSHOT_DIR, 'qliphoth', `${page.name}-section-${i}.png`),
              });
            }
          } catch (e) {
            // Section might not be screenshottable
          }
        }
      } catch (e) {
        console.log(`Failed to capture ${page.name}: ${e.message}`);
      }
    });
  }
});
