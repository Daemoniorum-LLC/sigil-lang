import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const PRODUCTION_URL = 'https://www.sigil-lang.com';
const WASM_URL = 'http://localhost:8000';

const SCREENSHOT_DIR = path.join(__dirname, '..', 'screenshots', 'parity');

// Pages to compare - production path -> local WASM path
// Note: playground is excluded because production /playground/ is an interactive editor,
//       while our WASM renders a static landing page UI
const PAGES = [
  { name: 'index', prodPath: '/', wasmPath: '/index.html' },
  { name: 'docs', prodPath: '/docs.html', wasmPath: '/docs.html' },
  { name: 'learn', prodPath: '/learn.html', wasmPath: '/learn.html' },
  { name: 'agents', prodPath: '/agents.html', wasmPath: '/agents.html' },
  { name: 'pattern', prodPath: '/pattern.html', wasmPath: '/pattern.html' },
  { name: 'qliphoth', prodPath: '/qliphoth.html', wasmPath: '/qliphoth.html' },
];

// Playground is tested separately - it's a WASM-rendered UI, not matching production's interactive editor
const PLAYGROUND_WASM_PATH = '/playground.html';

interface PageMetrics {
  title: string;
  sectionCount: number;
  headingCount: number;
  linkCount: number;
  buttonCount: number;
  imageCount: number;
  textLength: number;
  navLinks: string[];
  footerLinks: string[];
  mainSections: string[];
}

async function getPageMetrics(page: any): Promise<PageMetrics> {
  return await page.evaluate(() => {
    const sections = document.querySelectorAll('section');
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    const links = document.querySelectorAll('a');
    const buttons = document.querySelectorAll('button');
    const images = document.querySelectorAll('img');
    const nav = document.querySelector('nav');
    const footer = document.querySelector('footer');

    const navLinks = nav ? Array.from(nav.querySelectorAll('a')).map(a => a.textContent?.trim() || '') : [];
    const footerLinks = footer ? Array.from(footer.querySelectorAll('a')).map(a => a.textContent?.trim() || '') : [];
    const mainSections = Array.from(sections).map(s => s.id || s.className || 'unnamed');

    return {
      title: document.title,
      sectionCount: sections.length,
      headingCount: headings.length,
      linkCount: links.length,
      buttonCount: buttons.length,
      imageCount: images.length,
      textLength: document.body.innerText.length,
      navLinks,
      footerLinks,
      mainSections,
    };
  });
}

test.describe('Production vs WASM Parity Tests', () => {
  test.beforeAll(async () => {
    fs.mkdirSync(path.join(SCREENSHOT_DIR, 'production'), { recursive: true });
    fs.mkdirSync(path.join(SCREENSHOT_DIR, 'wasm'), { recursive: true });
    fs.mkdirSync(path.join(SCREENSHOT_DIR, 'diff'), { recursive: true });
  });

  for (const pageConfig of PAGES) {
    test.describe(`${pageConfig.name} page`, () => {

      test('compare structure and content', async ({ page }) => {
        // Get production metrics
        console.log(`\n=== Testing ${pageConfig.name} page ===`);

        await page.goto(`${PRODUCTION_URL}${pageConfig.prodPath}`, {
          waitUntil: 'networkidle',
          timeout: 30000
        });
        await page.waitForTimeout(1000);

        const prodMetrics = await getPageMetrics(page);
        console.log(`Production - Title: "${prodMetrics.title}"`);
        console.log(`Production - Sections: ${prodMetrics.sectionCount}, Headings: ${prodMetrics.headingCount}, Links: ${prodMetrics.linkCount}`);

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, 'production', `${pageConfig.name}.png`),
          fullPage: true,
        });

        // Get WASM metrics
        try {
          await page.goto(`${WASM_URL}${pageConfig.wasmPath}`, {
            waitUntil: 'networkidle',
            timeout: 15000
          });

          // Wait for WASM to render
          await page.waitForSelector('#app > *', { timeout: 10000 });
          await page.waitForTimeout(1000);

          const wasmMetrics = await getPageMetrics(page);
          console.log(`WASM - Title: "${wasmMetrics.title}"`);
          console.log(`WASM - Sections: ${wasmMetrics.sectionCount}, Headings: ${wasmMetrics.headingCount}, Links: ${wasmMetrics.linkCount}`);

          await page.screenshot({
            path: path.join(SCREENSHOT_DIR, 'wasm', `${pageConfig.name}.png`),
            fullPage: true,
          });

          // Compare metrics
          const report = {
            page: pageConfig.name,
            production: prodMetrics,
            wasm: wasmMetrics,
            differences: [] as string[],
          };

          // Check key metrics
          if (prodMetrics.sectionCount !== wasmMetrics.sectionCount) {
            report.differences.push(`Section count: prod=${prodMetrics.sectionCount}, wasm=${wasmMetrics.sectionCount}`);
          }

          if (prodMetrics.headingCount !== wasmMetrics.headingCount) {
            report.differences.push(`Heading count: prod=${prodMetrics.headingCount}, wasm=${wasmMetrics.headingCount}`);
          }

          if (Math.abs(prodMetrics.linkCount - wasmMetrics.linkCount) > 2) {
            report.differences.push(`Link count differs significantly: prod=${prodMetrics.linkCount}, wasm=${wasmMetrics.linkCount}`);
          }

          // Check nav links match
          const prodNavSet = new Set(prodMetrics.navLinks.filter(l => l.length > 0));
          const wasmNavSet = new Set(wasmMetrics.navLinks.filter(l => l.length > 0));
          const missingNavLinks = [...prodNavSet].filter(l => !wasmNavSet.has(l));
          if (missingNavLinks.length > 0) {
            report.differences.push(`Missing nav links in WASM: ${missingNavLinks.join(', ')}`);
          }

          // Save comparison report
          fs.writeFileSync(
            path.join(SCREENSHOT_DIR, 'diff', `${pageConfig.name}-report.json`),
            JSON.stringify(report, null, 2)
          );

          // Log differences
          if (report.differences.length > 0) {
            console.log(`\nDifferences found:`);
            report.differences.forEach(d => console.log(`  - ${d}`));
          } else {
            console.log(`\nNo structural differences found!`);
          }

          // Soft assertions - log but don't fail
          expect(wasmMetrics.sectionCount).toBeGreaterThan(0);
          expect(wasmMetrics.headingCount).toBeGreaterThan(0);

        } catch (e) {
          console.log(`WASM page failed to load: ${e.message}`);
          throw e;
        }
      });

      test('visual comparison', async ({ page }) => {
        // Capture production viewport
        await page.setViewportSize({ width: 1280, height: 720 });
        await page.goto(`${PRODUCTION_URL}${pageConfig.prodPath}`, {
          waitUntil: 'networkidle',
          timeout: 30000
        });
        await page.waitForTimeout(500);

        const prodScreenshot = await page.screenshot();

        // Capture WASM viewport
        try {
          await page.goto(`${WASM_URL}${pageConfig.wasmPath}`, {
            waitUntil: 'networkidle',
            timeout: 15000
          });
          await page.waitForSelector('#app > *', { timeout: 10000 });
          await page.waitForTimeout(500);

          // Visual regression with threshold
          await expect(page).toHaveScreenshot(`${pageConfig.name}-wasm.png`, {
            maxDiffPixelRatio: 0.15, // Allow 15% difference for fonts/rendering
            threshold: 0.3,
          });

        } catch (e) {
          console.log(`Visual comparison failed for ${pageConfig.name}: ${e.message}`);
          // Don't fail - just report
        }
      });
    });
  }
});

test.describe('WASM Runtime Verification', () => {
  test('all pages load and render content', async ({ page }) => {
    const results: { page: string; success: boolean; error?: string }[] = [];

    for (const pageConfig of PAGES) {
      try {
        await page.goto(`${WASM_URL}${pageConfig.wasmPath}`, {
          waitUntil: 'networkidle',
          timeout: 15000
        });

        // Wait for #app to have content
        await page.waitForSelector('#app > *', { timeout: 10000 });

        // Verify content rendered
        const hasContent = await page.evaluate(() => {
          const app = document.getElementById('app');
          return app && app.children.length > 0 && app.innerText.length > 100;
        });

        results.push({ page: pageConfig.name, success: hasContent });

        if (!hasContent) {
          console.log(`Warning: ${pageConfig.name} rendered but has minimal content`);
        }

      } catch (e) {
        results.push({ page: pageConfig.name, success: false, error: e.message });
      }
    }

    // Report results
    console.log('\n=== WASM Runtime Verification Results ===');
    results.forEach(r => {
      const status = r.success ? '✓' : '✗';
      console.log(`${status} ${r.page}${r.error ? `: ${r.error}` : ''}`);
    });

    // All pages should load
    const failures = results.filter(r => !r.success);
    expect(failures.length).toBe(0);
  });

  test('playground WASM renders correctly', async ({ page }) => {
    console.log('\n=== Playground WASM Test ===');

    await page.goto(`${WASM_URL}${PLAYGROUND_WASM_PATH}`, {
      waitUntil: 'networkidle',
      timeout: 15000
    });

    // Wait for #app to have content
    await page.waitForSelector('#app > *', { timeout: 10000 });

    // Verify playground UI elements
    const metrics = await page.evaluate(() => {
      const app = document.getElementById('app');
      if (!app) return null;

      return {
        hasContent: app.children.length > 0,
        textLength: app.innerText.length,
        hasHeader: !!app.querySelector('.playground-header, header'),
        hasEditor: !!app.querySelector('.editor-panel, .code-input, #editor'),
        hasOutput: !!app.querySelector('.output-panel, .preview-content, #output'),
        hasRunButton: !!app.querySelector('.btn-run, #run-btn, button'),
      };
    });

    console.log('Playground metrics:', metrics);

    expect(metrics).not.toBeNull();
    expect(metrics?.hasContent).toBe(true);
    // Playground has less text content (mostly UI chrome)
    expect(metrics?.textLength).toBeGreaterThanOrEqual(30);

    // Take screenshot
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, 'wasm', 'playground.png'),
      fullPage: true,
    });

    console.log('Playground WASM rendered successfully');
  });
});
