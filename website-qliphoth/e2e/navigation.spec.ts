import { test, expect } from '@playwright/test';

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('nav links are present', async ({ page }) => {
    const nav = page.locator('nav');
    await expect(nav.locator('a[href="docs.html"]')).toBeVisible();
    await expect(nav.locator('a[href="learn.html"]')).toBeVisible();
    await expect(nav.locator('a[href="agents.html"]')).toBeVisible();
    await expect(nav.locator('a[href="pattern.html"]')).toBeVisible();
  });

  test('logo links to home', async ({ page }) => {
    const logo = page.locator('.logo');
    await expect(logo).toHaveAttribute('href', '/');
  });

  test('nav link to docs works', async ({ page }) => {
    await page.click('nav a[href="docs.html"]');
    await expect(page).toHaveURL(/docs\.html/);
  });

  test('nav link to learn works', async ({ page }) => {
    await page.click('nav a[href="learn.html"]');
    await expect(page).toHaveURL(/learn\.html/);
  });

  test('nav link to agents works', async ({ page }) => {
    await page.click('nav a[href="agents.html"]');
    await expect(page).toHaveURL(/agents\.html/);
  });

  test('nav link to pattern works', async ({ page }) => {
    await page.click('nav a[href="pattern.html"]');
    await expect(page).toHaveURL(/pattern\.html/);
  });

  test('page card links work', async ({ page }) => {
    const docsCard = page.locator('.page-card[href="docs.html"]');
    await docsCard.click();
    await expect(page).toHaveURL(/docs\.html/);
  });
});

test.describe('CTA Buttons', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('primary CTA links to docs', async ({ page }) => {
    const primaryBtn = page.locator('.btn-primary');
    await expect(primaryBtn).toHaveAttribute('href', 'docs.html');
  });

  test('secondary CTA links to learn', async ({ page }) => {
    const secondaryBtn = page.locator('.btn-secondary');
    await expect(secondaryBtn).toHaveAttribute('href', 'learn.html');
  });
});
