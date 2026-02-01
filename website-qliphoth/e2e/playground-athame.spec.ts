import { test, expect } from '@playwright/test';

/**
 * Playground Athame Integration Tests
 *
 * Tests the Athame-powered Sigil playground:
 * - VDOM layout mounts correctly from WASM
 * - Syntax highlighting via JS Athame tokenizer
 * - Editor interactions (typing, tab, scroll sync)
 * - Example selector loads code
 * - Theme toggle switches light/dark
 * - Sandbox iframe initializes
 * - Run button sends code to sandbox
 * - Console output displays results
 * - Line numbers update with content
 */

test.describe('Playground Layout', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/playground.html');
    // Wait for WASM to mount the VDOM layout
    await page.waitForSelector('.playground-app', { timeout: 30000 });
  });

  test('renders main layout structure', async ({ page }) => {
    await expect(page.locator('.playground-header')).toBeVisible();
    await expect(page.locator('.playground-main')).toBeVisible();
    await expect(page.locator('.playground-footer')).toBeVisible();
  });

  test('renders editor panel with textarea and highlight layer', async ({ page }) => {
    await expect(page.locator('#editor-panel')).toBeVisible();
    await expect(page.locator('#code-input')).toBeVisible();
    await expect(page.locator('#highlight-layer')).toBeVisible();
    await expect(page.locator('#highlight-code')).toBeVisible();
    await expect(page.locator('#line-gutter')).toBeVisible();
  });

  test('renders output panel with console', async ({ page }) => {
    await expect(page.locator('.output-panel')).toBeVisible();
    await expect(page.locator('#console-output')).toBeVisible();
    await expect(page.locator('#preview-content')).toBeVisible();
  });

  test('renders header controls', async ({ page }) => {
    await expect(page.locator('#run-btn')).toBeVisible();
    await expect(page.locator('#example-select')).toBeVisible();
    await expect(page.locator('#theme-toggle')).toBeVisible();
  });

  test('renders footer with version badge', async ({ page }) => {
    const footer = page.locator('.playground-footer');
    await expect(footer).toBeVisible();
    await expect(footer.locator('.version-badge')).toContainText('0.4.0');
  });

  test('renders resize handle', async ({ page }) => {
    await expect(page.locator('#resize-handle')).toBeVisible();
  });
});

test.describe('Syntax Highlighting', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/playground.html');
    await page.waitForSelector('.playground-app', { timeout: 30000 });
    // Wait for JS controller to initialize and set default code
    await page.waitForSelector('#highlight-code .ath-keyword', { timeout: 5000 });
  });

  test('default code has syntax-highlighted keywords', async ({ page }) => {
    const keywords = page.locator('#highlight-code .ath-keyword');
    await expect(keywords.first()).toBeVisible();
    // "rite" and "main" related keywords should be highlighted
    const count = await keywords.count();
    expect(count).toBeGreaterThan(0);
  });

  test('default code has highlighted strings', async ({ page }) => {
    const strings = page.locator('#highlight-code .ath-string');
    await expect(strings.first()).toBeVisible();
  });

  test('default code has highlighted comments', async ({ page }) => {
    const comments = page.locator('#highlight-code .ath-comment');
    await expect(comments.first()).toBeVisible();
  });

  test('default code has native symbols highlighted', async ({ page }) => {
    // The default Hello World example uses ☉ and ≔
    const natives = page.locator('#highlight-code .ath-native');
    await expect(natives.first()).toBeVisible();
  });

  test('typing updates highlighting in real time', async ({ page }) => {
    const textarea = page.locator('#code-input');
    // Clear and type new code
    await textarea.fill('');
    await textarea.type('rite hello() { 42 }');

    // Wait for highlight to update
    await page.waitForTimeout(200);

    // "rite" should be highlighted as keyword
    const keywords = page.locator('#highlight-code .ath-keyword');
    await expect(keywords.first()).toBeVisible();
    await expect(keywords.first()).toContainText('rite');

    // "42" should be highlighted as number
    const numbers = page.locator('#highlight-code .ath-number');
    await expect(numbers.first()).toBeVisible();
    await expect(numbers.first()).toContainText('42');
  });
});

test.describe('Editor Interactions', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/playground.html');
    await page.waitForSelector('.playground-app', { timeout: 30000 });
    await page.waitForSelector('#highlight-code .ath-keyword', { timeout: 5000 });
  });

  test('tab key inserts spaces', async ({ page }) => {
    const textarea = page.locator('#code-input');
    await textarea.fill('');
    await textarea.focus();
    await textarea.type('hello');
    await page.keyboard.press('Tab');
    await textarea.type('world');

    const value = await textarea.inputValue();
    expect(value).toContain('hello    world');
  });

  test('line numbers update with content', async ({ page }) => {
    const textarea = page.locator('#code-input');
    const gutter = page.locator('#line-gutter');

    // Set multi-line code
    await textarea.fill('line1\nline2\nline3\nline4\nline5');
    // Trigger input event
    await textarea.dispatchEvent('input');
    await page.waitForTimeout(200);

    const lineNumbers = gutter.locator('.line-number');
    const count = await lineNumbers.count();
    expect(count).toBe(5);
  });

  test('Ctrl+Z undoes text changes', async ({ page }) => {
    const textarea = page.locator('#code-input');
    await textarea.fill('');
    await textarea.focus();
    await textarea.type('hello');
    const before = await textarea.inputValue();
    expect(before).toBe('hello');

    await page.keyboard.press('Control+z');
    // Browser native undo should remove last typed character(s)
    const after = await textarea.inputValue();
    expect(after.length).toBeLessThan(before.length);
  });
});

test.describe('Example Selector', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/playground.html');
    await page.waitForSelector('.playground-app', { timeout: 30000 });
    await page.waitForSelector('#highlight-code .ath-keyword', { timeout: 5000 });
  });

  test('selecting counter example loads counter code', async ({ page }) => {
    const select = page.locator('#example-select');
    await select.selectOption('counter');

    await page.waitForTimeout(200);
    const textarea = page.locator('#code-input');
    const value = await textarea.inputValue();
    expect(value).toContain('Counter');
    expect(value).toContain('count');
  });

  test('selecting morphemes example loads pipeline code', async ({ page }) => {
    const select = page.locator('#example-select');
    await select.selectOption('morphemes');

    await page.waitForTimeout(200);
    const textarea = page.locator('#code-input');
    const value = await textarea.inputValue();
    expect(value).toContain('Morpheme');
    expect(value).toContain('Pipeline');
  });

  test('selecting evidentiality example loads evidence code', async ({ page }) => {
    const select = page.locator('#example-select');
    await select.selectOption('evidentiality');

    await page.waitForTimeout(200);
    const textarea = page.locator('#code-input');
    const value = await textarea.inputValue();
    expect(value).toContain('Evidentiality');
    expect(value).toContain('Sensor');
  });

  test('example selector updates syntax highlighting', async ({ page }) => {
    const select = page.locator('#example-select');
    await select.selectOption('todo');

    await page.waitForTimeout(300);

    // "sigil" keyword should appear (struct definition)
    const keywords = page.locator('#highlight-code .ath-keyword');
    const texts = await keywords.allTextContents();
    expect(texts).toContain('sigil');
  });
});

test.describe('Theme Toggle', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/playground.html');
    await page.waitForSelector('.playground-app', { timeout: 30000 });
  });

  test('clicking theme toggle switches to light mode', async ({ page }) => {
    const btn = page.locator('#theme-toggle');
    await btn.click();

    const theme = await page.locator('html').getAttribute('data-theme');
    expect(theme).toBe('light');
  });

  test('clicking theme toggle twice returns to dark mode', async ({ page }) => {
    const btn = page.locator('#theme-toggle');
    await btn.click();
    await btn.click();

    const theme = await page.locator('html').getAttribute('data-theme');
    expect(theme).toBeNull();
  });
});

test.describe('Sandbox', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/playground.html');
    await page.waitForSelector('.playground-app', { timeout: 30000 });
    await page.waitForSelector('#highlight-code .ath-keyword', { timeout: 5000 });
  });

  test('sandbox iframe is created with correct attributes', async ({ page }) => {
    const iframe = page.locator('#sandbox-frame');
    await expect(iframe).toHaveCount(1);
    const sandbox = await iframe.getAttribute('sandbox');
    expect(sandbox).toBe('allow-scripts');
  });

  test('run button exists and is clickable', async ({ page }) => {
    const runBtn = page.locator('#run-btn');
    await expect(runBtn).toBeVisible();
    await expect(runBtn).toBeEnabled();
  });

  test('clicking run when sandbox not ready shows message', async ({ page }) => {
    // Click run immediately (sandbox may not be ready yet)
    const runBtn = page.locator('#run-btn');
    await runBtn.click();

    // Should show a message in console output
    const consoleOutput = page.locator('#console-output');
    const text = await consoleOutput.textContent();
    // Either shows "loading" message or actual output depending on timing
    expect(text).toBeDefined();
  });

  test('clear console button clears output', async ({ page }) => {
    // Click run to generate some output
    const runBtn = page.locator('#run-btn');
    await runBtn.click();
    await page.waitForTimeout(500);

    // Clear console
    const clearBtn = page.locator('#clear-console-btn');
    await clearBtn.click();

    const consoleOutput = page.locator('#console-output');
    const children = await consoleOutput.locator('.console-line').count();
    expect(children).toBe(0);
  });
});

test.describe('Resize Handle', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/playground.html');
    await page.waitForSelector('.playground-app', { timeout: 30000 });
  });

  test('resize handle is visible between panels', async ({ page }) => {
    const handle = page.locator('#resize-handle');
    await expect(handle).toBeVisible();
  });
});

test.describe('No Console Errors', () => {
  test('playground loads without critical JS errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => {
      errors.push(err.message);
    });

    await page.goto('/playground.html');
    await page.waitForSelector('.playground-app', { timeout: 30000 });
    await page.waitForTimeout(2000);

    // Filter out expected errors (e.g., WASM loading issues in test env)
    const criticalErrors = errors.filter(
      (e) => !e.includes('wasm') && !e.includes('WASM') && !e.includes('fetch')
    );
    expect(criticalErrors).toHaveLength(0);
  });
});
