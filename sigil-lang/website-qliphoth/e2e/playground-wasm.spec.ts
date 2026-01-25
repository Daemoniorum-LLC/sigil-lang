import { test, expect } from '@playwright/test';

test.describe('Playground WASM Integration', () => {
  test.beforeEach(async ({ page }) => {
    // Go to the playground
    await page.goto('http://localhost:5173/');
    // Wait for WASM to load
    await page.waitForSelector('text=Sigil runtime ready', { timeout: 30000 });
  });

  test('should load WASM interpreter and show WASM mode', async ({ page }) => {
    // Check that WASM mode is active (not Backend, not Mock)
    const output = page.locator('#output');
    await expect(output).toContainText('WASM mode');
  });

  test('should execute simple program and return result', async ({ page }) => {
    // Clear editor and type canonical Sigil code
    const editor = page.locator('.cm-content');
    await editor.click();
    await page.keyboard.press('Control+a');
    await page.keyboard.type('rite main() → i64 { 42 }');

    // Click Run button
    await page.click('#run');

    // Wait for output
    await page.waitForTimeout(1000);

    // Check for actual execution result
    const output = page.locator('#output');
    await expect(output).toContainText('42');
    await expect(output).toContainText('Completed');
  });

  test('should execute arithmetic with let bindings', async ({ page }) => {
    const editor = page.locator('.cm-content');
    await editor.click();
    await page.keyboard.press('Control+a');
    await page.keyboard.type('rite main() → i64 { ≔ x = 10; ≔ y = 20; x + y }');

    await page.click('#run');
    await page.waitForTimeout(1000);

    const output = page.locator('#output');
    await expect(output).toContainText('30');
  });

  test('should execute with struct field access', async ({ page }) => {
    const editor = page.locator('.cm-content');
    await editor.click();
    await page.keyboard.press('Control+a');
    await page.keyboard.type('sigil Point { x: i64, y: i64 } rite main() → i64 { ≔ p = Point { x: 1, y: 2 }; p.x }');

    await page.click('#run');
    await page.waitForTimeout(1000);

    const output = page.locator('#output');
    await expect(output).toContainText('1');
  });

  test('should report parse errors for invalid syntax', async ({ page }) => {
    const editor = page.locator('.cm-content');
    await editor.click();
    await page.keyboard.press('Control+a');
    // Malformed code - missing function body close
    await page.keyboard.type('rite main() → i64 { @@@invalid@@@ }');

    await page.click('#run');
    await page.waitForTimeout(1000);

    const output = page.locator('#output');
    // Should show an error (case insensitive check)
    await expect(output).toContainText(/error/i);
  });

  test('should handle Check button for syntax validation', async ({ page }) => {
    const editor = page.locator('.cm-content');
    await editor.click();
    await page.keyboard.press('Control+a');
    await page.keyboard.type('rite add(a: i64, b: i64) → i64 { a + b } rite main() → i64 { add(1, 2) }');

    // Click Check button instead of Run
    await page.click('#check');
    await page.waitForTimeout(1000);

    const output = page.locator('#output');
    await expect(output).toContainText('passed');
  });

  test('should execute enum and return zero', async ({ page }) => {
    const editor = page.locator('.cm-content');
    await editor.click();
    await page.keyboard.press('Control+a');
    await page.keyboard.type('ᛈ Color { Red, Green, Blue } rite main() → i64 { 0 }');

    await page.click('#run');
    await page.waitForTimeout(1000);

    const output = page.locator('#output');
    await expect(output).toContainText('0');
  });

  test('should execute impl methods', async ({ page }) => {
    const editor = page.locator('.cm-content');
    await editor.click();
    await page.keyboard.press('Control+a');
    await page.keyboard.type('sigil Foo { val: i64 } ⊢ Foo { rite get(self) → i64 { self.val } } rite main() → i64 { 0 }');

    await page.click('#run');
    await page.waitForTimeout(1000);

    const output = page.locator('#output');
    await expect(output).toContainText('0');
  });

  test('should execute if-else expressions', async ({ page }) => {
    const editor = page.locator('.cm-content');
    await editor.click();
    await page.keyboard.press('Control+a');
    await page.keyboard.type('rite main() → i64 { if true { 42 } else { 0 } }');

    await page.click('#run');
    await page.waitForTimeout(1000);

    const output = page.locator('#output');
    await expect(output).toContainText('42');
  });

  test('should execute match expressions', async ({ page }) => {
    const editor = page.locator('.cm-content');
    await editor.click();
    await page.keyboard.press('Control+a');
    await page.keyboard.type('rite main() → i64 { ≔ x = 3; match x { 1 => 10, 2 => 20, 3 => 30, _ => 0 } }');

    await page.click('#run');
    await page.waitForTimeout(1000);

    const output = page.locator('#output');
    await expect(output).toContainText('30');
  });

  test('should capture println output', async ({ page }) => {
    const editor = page.locator('.cm-content');
    await editor.click();
    await page.keyboard.press('Control+a');
    await page.keyboard.type('rite main() → i64 { println("Hello, Sigil!"); 0 }');

    await page.click('#run');
    await page.waitForTimeout(1000);

    const output = page.locator('#output');
    await expect(output).toContainText('Hello, Sigil!');
  });
});
