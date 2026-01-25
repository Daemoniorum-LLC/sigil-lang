import { test, expect } from '@playwright/test';

test.describe('Smoke Tests', () => {
  test('homepage loads', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/Sigil/);
  });

  test('homepage has hero section', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.hero h1')).toBeVisible();
    await expect(page.locator('.hero h1')).toContainText('Polysynthetic');
  });

  test('homepage has navigation', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('nav')).toBeVisible();
  });

  test('homepage has CTA buttons', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.cta-buttons .btn-primary')).toBeVisible();
    await expect(page.locator('.cta-buttons .btn-secondary')).toBeVisible();
  });

  test('homepage has feature cards', async ({ page }) => {
    await page.goto('/');
    const featureCards = page.locator('.feature-card');
    await expect(featureCards).toHaveCount(4);
  });

  test('homepage has page cards', async ({ page }) => {
    await page.goto('/');
    const pageCards = page.locator('.page-card');
    await expect(pageCards.first()).toBeVisible();
  });

  test('homepage has code preview', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.code-preview')).toBeVisible();
    await expect(page.locator('.code-block')).toBeVisible();
  });

  test('homepage has footer', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('footer')).toBeVisible();
    await expect(page.locator('footer')).toContainText('Daemoniorum');
  });
});

test.describe('Existing WASM Pages', () => {
  test('docs page loads', async ({ page }) => {
    await page.goto('/docs.html');
    await expect(page).toHaveTitle(/Sigil|Docs/);
  });

  test('learn page loads', async ({ page }) => {
    await page.goto('/learn.html');
    await expect(page).toHaveTitle(/Sigil|Learn/);
  });

  test('agents page loads', async ({ page }) => {
    await page.goto('/agents.html');
    await expect(page).toHaveTitle(/Sigil|Agents/);
  });

  test('pattern page loads', async ({ page }) => {
    await page.goto('/pattern.html');
    await expect(page).toHaveTitle(/Sigil|Pattern/);
  });
});

test.describe('WASM Runtime', () => {
  test('sigil_runtime.js is accessible', async ({ page }) => {
    const response = await page.goto('/sigil_runtime.js');
    expect(response?.status()).toBe(200);
  });

  test('WASM files are accessible', async ({ page }) => {
    // Check that at least one WASM file exists
    const response = await page.goto('/docs.wasm');
    expect(response?.status()).toBe(200);
  });
});
