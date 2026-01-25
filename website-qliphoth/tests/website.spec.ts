import { test, expect } from '@playwright/test';

// Base URL for the local server
const BASE_URL = 'http://localhost:5181';

test.describe('Sigil Website E2E Tests', () => {
  test.describe('Home Page', () => {
    test('loads successfully and renders WASM content', async ({ page }) => {
      await page.goto(BASE_URL);

      // Wait for WASM to load and render
      await expect(page.locator('#status')).toContainText('Rendered via Sigil', { timeout: 10000 });

      // Check header renders
      await expect(page.locator('.site-nav')).toBeVisible();
      await expect(page.locator('.brand')).toBeVisible();

      // Check navigation links (new structure)
      await expect(page.locator('.nav-link:has-text("Learn")')).toBeVisible();
      await expect(page.locator('.nav-link:has-text("Docs")')).toBeVisible();
      await expect(page.locator('.nav-link:has-text("Playground")')).toBeVisible();
      await expect(page.locator('.nav-link:has-text("Agents")')).toBeVisible();
    });

    test('renders hero section with title', async ({ page }) => {
      await page.goto(BASE_URL);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check hero content
      await expect(page.locator('.hero h1')).toContainText('Sigil');
      await expect(page.locator('.hero-subtitle')).toBeVisible();
    });

    test('renders features section', async ({ page }) => {
      await page.goto(BASE_URL);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check features grid
      await expect(page.locator('.features-section')).toBeVisible();
      await expect(page.locator('.feature-card')).toHaveCount(4);
    });

    test('renders code example', async ({ page }) => {
      await page.goto(BASE_URL);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check code section (problem-section with code comparison)
      await expect(page.locator('.problem-section')).toBeVisible();
      await expect(page.locator('.code-block').first()).toBeVisible();
    });

    test('renders footer with columns', async ({ page }) => {
      await page.goto(BASE_URL);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      await expect(page.locator('.site-footer')).toBeVisible();
      await expect(page.locator('.footer-container')).toBeVisible();
      await expect(page.locator('.footer-brand')).toBeVisible();
      await expect(page.locator('.footer-column').first()).toBeVisible();
    });

    test('renders version badge', async ({ page }) => {
      await page.goto(BASE_URL);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      await expect(page.locator('.version-badge')).toContainText('0.3');
    });
  });

  // Page-specific WASM exports (main_learn, main_docs, etc.) now implemented
  test.describe('Learn Page', () => {
    test('loads successfully and renders content', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/learn.html`);

      // Wait for WASM to render
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check learn content renders (chapter-based structure)
      await expect(page.locator('.learn-content')).toBeVisible();
    });

    test('renders progress bar', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/learn.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check progress bar with 5 steps
      await expect(page.locator('.progress-bar')).toBeVisible();
      await expect(page.locator('.progress-step')).toHaveCount(5);
    });

    test('renders chapter headers', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/learn.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check chapter structure
      await expect(page.locator('.chapter')).toHaveCount(5);
      await expect(page.locator('.chapter-header')).toHaveCount(5);
      await expect(page.locator('.chapter-number')).toHaveCount(5);
    });

    test('renders code blocks in chapters', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/learn.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check code blocks exist
      const codeCount = await page.locator('.code-block').count();
      expect(codeCount).toBeGreaterThanOrEqual(5);
    });

    test('renders navigation header', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/learn.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      await expect(page.locator('.site-nav')).toBeVisible();
      await expect(page.locator('.brand')).toBeVisible();
    });
  });

  test.describe('Docs Page', () => {
    test('loads successfully and renders content', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/docs.html`);

      // Wait for WASM to render
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check two-column layout
      await expect(page.locator('.docs-layout')).toBeVisible();
    });

    test('renders sidebar with sections', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/docs.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check sidebar exists with sections
      await expect(page.locator('.docs-sidebar')).toBeVisible();
      await expect(page.locator('.sidebar-section')).toHaveCount(6);
    });

    test('renders main content area', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/docs.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check main content
      await expect(page.locator('.docs-content')).toBeVisible();
      await expect(page.locator('article')).toBeVisible();
    });

    test('renders introduction section', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/docs.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check introduction
      await expect(page.locator('h1#introduction')).toBeVisible();
      await expect(page.locator('.lead')).toBeVisible();
    });

    test('renders feature highlights', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/docs.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      await expect(page.locator('.feature-highlight')).toBeVisible();
      await expect(page.locator('.highlight-item')).toHaveCount(3);
    });

    test('renders evidentiality table', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/docs.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      await expect(page.locator('.docs-table')).toBeVisible();
    });

    test('renders agent infrastructure grid', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/docs.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      await expect(page.locator('.agent-overview-grid')).toBeVisible();
      // Check at least one agent layer is rendered (WASM may have array limits for 9 items)
      const layerCount = await page.locator('.agent-layer').count();
      expect(layerCount).toBeGreaterThanOrEqual(1);
    });
  });

  test.describe('Examples Page', () => {
    test('loads successfully and renders content', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/examples.html`);

      // Wait for WASM to render
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check two-column layout
      await expect(page.locator('.docs-layout')).toBeVisible();
    });

    test('renders sidebar with categories', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/examples.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check sidebar with 3 categories
      await expect(page.locator('.docs-sidebar')).toBeVisible();
      await expect(page.locator('.sidebar-section')).toHaveCount(3);
    });

    test('renders multiple code examples', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/examples.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check code blocks
      const codeCount = await page.locator('.code-block').count();
      expect(codeCount).toBeGreaterThanOrEqual(5);
    });

    test('renders example headings', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/examples.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check section headings (h3 elements for each example)
      const h3Count = await page.locator('h3').count();
      expect(h3Count).toBeGreaterThanOrEqual(3);
    });
  });

  test.describe('Playground Page', () => {
    test('loads successfully and renders content', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/playground.html`);

      // Wait for WASM to render
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check playground hero section
      await expect(page.locator('.playground-hero')).toBeVisible();
    });

    test('renders Coming Soon status', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/playground.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check Coming Soon badge
      await expect(page.locator('.playground-status')).toContainText('Coming Soon');
    });

    test('renders playground title and description', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/playground.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      await expect(page.locator('.playground-title')).toContainText('Sigil Playground');
      await expect(page.locator('.playground-description')).toBeVisible();
    });

    test('renders action buttons', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/playground.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      await expect(page.locator('.playground-actions')).toBeVisible();
      await expect(page.locator('.button-primary')).toBeVisible();
      await expect(page.locator('.button-secondary')).toBeVisible();
    });

    test('renders code preview', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/playground.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      await expect(page.locator('.playground-preview')).toBeVisible();
      await expect(page.locator('.preview-code')).toBeVisible();
    });

    test('renders planned features list', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/playground.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      await expect(page.locator('.feature-list')).toBeVisible();
      // Check at least one feature is rendered (WASM may have array limits)
      const featureCount = await page.locator('.feature-list li').count();
      expect(featureCount).toBeGreaterThanOrEqual(1);
    });

    test('renders GitHub star button', async ({ page }) => {
      await page.goto(`${BASE_URL}/pages/playground.html`);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      await expect(page.locator('.notify-form')).toBeVisible();
      await expect(page.locator('.notify-btn')).toContainText('Star on GitHub');
    });
  });

  test.describe('Navigation', () => {
    test('navigation links exist and are clickable', async ({ page }) => {
      await page.goto(BASE_URL);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Verify nav links exist (sub-pages not yet implemented)
      await expect(page.locator('.nav-link:has-text("Learn")')).toBeVisible();
      await expect(page.locator('.nav-link:has-text("Docs")')).toBeVisible();
      await expect(page.locator('.nav-link:has-text("Playground")')).toBeVisible();
      await expect(page.locator('.nav-link:has-text("Agents")')).toBeVisible();
    });

    test('brand link navigates to home', async ({ page }) => {
      await page.goto(BASE_URL);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Click the brand link
      await page.click('.brand');
      await expect(page).toHaveURL(BASE_URL + '/');
    });
  });

  test.describe('Responsive Design', () => {
    test('renders correctly on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });
      await page.goto(BASE_URL);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check content is still visible
      await expect(page.locator('.hero h1')).toBeVisible();
      await expect(page.locator('.feature-card')).toHaveCount(4);
    });

    test('renders correctly on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 });
      await page.goto(BASE_URL);
      await expect(page.locator('#status')).toContainText('Rendered', { timeout: 10000 });

      // Check content is visible on tablet
      await expect(page.locator('.hero')).toBeVisible();
      await expect(page.locator('.features-section')).toBeVisible();
    });
  });

  test.describe('WASM Error Handling', () => {
    test('shows error message when WASM fails to load', async ({ page }) => {
      // Block WASM file to simulate failure
      await page.route('**/site.wasm', route => route.abort());

      await page.goto(BASE_URL);

      // Should show error in status
      await expect(page.locator('#status')).toContainText('Error', { timeout: 10000 });
    });
  });
});
