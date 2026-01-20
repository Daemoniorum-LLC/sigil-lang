import { test } from '@playwright/test';

const EXAMPLES = [
  'hello',
  'variables',
  'functions',
  'pipes',
  'transform',
  'filter',
  'aggregate',
  'evidence',
  'validation',
  'structs',
  'matching',
  'pipeline'
];

test.describe('Playground Example Audit', () => {
  test('audit all examples', async ({ page }) => {
    await page.goto('https://sigil-lang.com/playground/');

    // Wait for page to load
    await page.waitForTimeout(3000);

    const results: { name: string; status: string; output: string }[] = [];

    for (const example of EXAMPLES) {
      console.log('\n=== Testing: ' + example + ' ===');

      // Select the example from dropdown
      const dropdown = page.locator('#examples, #example-select, select').first();
      if (await dropdown.isVisible()) {
        await dropdown.selectOption(example);
        await page.waitForTimeout(500);
      }

      // Click Run button
      const runBtn = page.locator('#run, #run-btn, button:has-text("Run")').first();
      if (await runBtn.isVisible()) {
        await runBtn.click();
      }

      // Wait for execution
      await page.waitForTimeout(2000);

      // Get output
      const output = await page.locator('#output, #console-output, .output, .console-output').first().textContent() || '';

      const hasError = output.toLowerCase().includes('error') ||
                       output.toLowerCase().includes('failed') ||
                       output.includes('Expected');

      const status = hasError ? 'FAIL' : 'PASS';

      console.log('Status: ' + status);
      console.log('Output: ' + output.slice(0, 500));

      results.push({ name: example, status, output: output.slice(0, 500) });
    }

    console.log('\n\n========== SUMMARY ==========');
    for (const r of results) {
      const icon = r.status === 'FAIL' ? 'FAIL' : 'PASS';
      console.log(icon + ' ' + r.name + ': ' + r.status);
      if (r.status === 'FAIL') {
        console.log('   Output: ' + r.output.slice(0, 300));
      }
    }

    const failures = results.filter(r => r.status === 'FAIL');
    console.log('\nTotal: ' + results.length + ' | Passed: ' + (results.length - failures.length) + ' | Failed: ' + failures.length);
  });
});
