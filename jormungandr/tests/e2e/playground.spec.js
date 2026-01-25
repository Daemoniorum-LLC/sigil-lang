/**
 * Sigil Playground E2E Tests
 * TDD Phase 4: Playground Integration
 *
 * Run with: npx playwright test tests/e2e/playground.spec.js
 */

const { test, expect } = require('@playwright/test');

test.describe('Sigil Playground with Jormungandr', () => {
    test.beforeEach(async ({ page }) => {
        // Navigate to playground
        await page.goto('/playground.html');

        // Wait for WASM to load (compiler becomes available)
        await page.waitForFunction(
            () => window.compiler !== undefined,
            { timeout: 10000 }
        );
    });

    // ========================================================================
    // Basic Compilation Tests
    // ========================================================================

    test('runs hello world', async ({ page }) => {
        // Clear and type code
        const editor = page.locator('#code-input');
        await editor.fill('rite main() -> i64 { println("Hello, Sigil!"); 0 }');

        // Click run
        await page.click('#run-btn');

        // Wait for execution
        await page.waitForSelector('.console-success', { timeout: 5000 });

        // Check output
        const output = await page.locator('#console-output').textContent();
        expect(output).toContain('Hello, Sigil!');
    });

    test('shows syntax errors', async ({ page }) => {
        const editor = page.locator('#code-input');
        await editor.fill('rite main( { }');
        await page.click('#run-btn');

        // Wait for error
        await page.waitForSelector('.console-error', { timeout: 5000 });

        const output = await page.locator('#console-output').textContent();
        expect(output.toLowerCase()).toContain('error');

        // Status should show error
        const statusDot = page.locator('#status-dot');
        await expect(statusDot).toHaveClass(/status-error/);
    });

    test('shows type errors', async ({ page }) => {
        const editor = page.locator('#code-input');
        await editor.fill('rite main() -> i64 { "not an int" }');
        await page.click('#run-btn');

        await page.waitForSelector('.console-error', { timeout: 5000 });

        const output = await page.locator('#console-output').textContent();
        expect(output.toLowerCase()).toContain('type');
    });

    test('handles runtime errors', async ({ page }) => {
        const editor = page.locator('#code-input');
        await editor.fill('rite main() -> i64 { let x = 1 / 0; println(str(x)); 0 }');
        await page.click('#run-btn');

        await page.waitForSelector('.console-error', { timeout: 5000 });

        const output = await page.locator('#console-output').textContent();
        expect(output.toLowerCase()).toContain('division');
    });

    // ========================================================================
    // Sigil Language Features
    // ========================================================================

    test('runs morpheme examples', async ({ page }) => {
        // Select morphemes example from dropdown
        await page.selectOption('#example-select', 'morphemes');
        await page.click('#run-btn');

        // Wait for success
        await page.waitForSelector('.console-success', { timeout: 5000 });

        // Status should show success
        const statusText = await page.locator('#status-text').textContent();
        expect(statusText).toBe('Success');
    });

    test('handles evidentiality markers', async ({ page }) => {
        const editor = page.locator('#code-input');
        await editor.fill(`
rite main() -> i64 {
    let answer! = 42;
    println(evidence_of(answer));
    0
}
        `);
        await page.click('#run-btn');

        await page.waitForSelector('.console-success', { timeout: 5000 });

        const output = await page.locator('#console-output').textContent();
        expect(output.toLowerCase()).toContain('known');
    });

    // ========================================================================
    // Keyboard Shortcuts
    // ========================================================================

    test('Ctrl+Enter runs code', async ({ page }) => {
        const editor = page.locator('#code-input');
        await editor.fill('rite main() -> i64 { println("shortcut"); 0 }');

        // Press Ctrl+Enter
        await editor.press('Control+Enter');

        await page.waitForSelector('.console-success', { timeout: 5000 });

        const output = await page.locator('#console-output').textContent();
        expect(output).toContain('shortcut');
    });

    test('Ctrl+S shares code', async ({ page }) => {
        const editor = page.locator('#code-input');
        await editor.fill('rite main() -> i64 { 42 }');

        // Press Ctrl+S
        await editor.press('Control+s');

        // Should show "copied" message
        const output = await page.locator('#console-output').textContent();
        expect(output.toLowerCase()).toContain('copied');
    });

    // ========================================================================
    // UI Features
    // ========================================================================

    test('clear console button works', async ({ page }) => {
        const editor = page.locator('#code-input');
        await editor.fill('rite main() -> i64 { println("test"); 0 }');
        await page.click('#run-btn');

        await page.waitForSelector('.console-success', { timeout: 5000 });

        // Clear console
        await page.click('#clear-console-btn');

        const output = await page.locator('#console-output').textContent();
        expect(output.trim()).toBe('');
    });

    test('theme toggle works', async ({ page }) => {
        // Click theme toggle
        await page.click('#theme-toggle');

        // Should be light theme
        const theme = await page.locator('html').getAttribute('data-theme');
        expect(theme).toBe('light');

        // Toggle back
        await page.click('#theme-toggle');

        const theme2 = await page.locator('html').getAttribute('data-theme');
        expect(theme2 || 'dark').toBe('dark');
    });

    test('share button generates URL', async ({ page }) => {
        const editor = page.locator('#code-input');
        await editor.fill('rite main() -> i64 { 42 }');

        await page.click('#share-btn');

        const output = await page.locator('#console-output').textContent();
        expect(output.toLowerCase()).toContain('copied');
    });

    test('loads code from URL hash', async ({ page }) => {
        const code = 'rite main() -> i64 { println("from url"); 0 }';
        const encoded = Buffer.from(encodeURIComponent(code)).toString('base64');

        await page.goto(`/playground.html#code=${encoded}`);
        await page.waitForFunction(() => window.compiler !== undefined, { timeout: 10000 });

        const editorContent = await page.locator('#code-input').inputValue();
        expect(editorContent).toContain('from url');
    });

    // ========================================================================
    // Performance
    // ========================================================================

    test('displays execution time', async ({ page }) => {
        const editor = page.locator('#code-input');
        await editor.fill('rite main() -> i64 { 0 }');
        await page.click('#run-btn');

        await page.waitForSelector('.console-info', { timeout: 5000 });

        const output = await page.locator('#console-output').textContent();
        expect(output).toMatch(/\d+(\.\d+)?ms/);
    });
});
