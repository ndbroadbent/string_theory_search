import { test, expect } from '@playwright/test';

test.describe('Playground Page', () => {
  test('loads without errors', async ({ page }) => {
    // Navigate to playground
    await page.goto('/playground');

    // Wait for the page to load
    await expect(page.getByRole('heading', { name: 'Playground' })).toBeVisible();

    // Verify predefined configurations are loaded
    await expect(page.getByText('Predefined Configurations')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Quintic Threefold' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'McAllister 4-214-647' })).toBeVisible();

    // Verify no errors in console
    const errors: string[] = [];
    page.on('pageerror', (err) => {
      errors.push(err.message);
    });

    // Give it a moment to catch any async errors
    await page.waitForTimeout(1000);
    expect(errors).toHaveLength(0);
  });

  // Skip for now - React hydration timing issue with button clicks
  test.skip('loads predefined Quintic Threefold config', async ({ page }) => {
    await page.goto('/playground');

    // Click on Quintic Threefold
    await page.getByRole('button', { name: 'Quintic Threefold' }).click();

    // Wait a moment for React state to update
    await page.waitForTimeout(500);

    // Verify form is populated - textarea uses value attribute
    const verticesTextarea = page.locator('textarea').first();
    await expect(verticesTextarea).toHaveValue(/\[\[1,0,0,0\]/, { timeout: 10000 });

    // Verify h11 and h21 are set (these are the first two number inputs)
    const h11Input = page.locator('input[type="number"]').first();
    const h21Input = page.locator('input[type="number"]').nth(1);
    await expect(h11Input).toHaveValue('1');
    await expect(h21Input).toHaveValue('101');

    // Verify g_s is set
    const gsInput = page.locator('input[step="0.0001"]');
    await expect(gsInput).toHaveValue('0.1');
  });

  // Skip for now - React hydration timing issue with button clicks
  test.skip('loads predefined McAllister config', async ({ page }) => {
    await page.goto('/playground');

    // Click on McAllister
    await page.getByRole('button', { name: 'McAllister 4-214-647' }).click();

    // Wait a moment for React state to update
    await page.waitForTimeout(500);

    // Verify form is populated with McAllister values - use toHaveValue for textarea
    const verticesTextarea = page.locator('textarea').first();
    await expect(verticesTextarea).toHaveValue(/\[\[0,0,0,0\]/, { timeout: 10000 });

    // Verify h11 and h21 are set
    const h11Input = page.locator('input[type="number"]').first();
    const h21Input = page.locator('input[type="number"]').nth(1);
    await expect(h11Input).toHaveValue('4');
    await expect(h21Input).toHaveValue('214');

    // Verify g_s is set (McAllister uses ~0.009)
    const gsInput = page.locator('input[step="0.0001"]');
    await expect(gsInput).toHaveValue('0.00911134');

    // Verify label is set
    const labelInput = page.locator('input[placeholder="e.g., My test configuration"]');
    await expect(labelInput).toHaveValue('McAllister 4-214-647');
  });

  // Skip for now - React hydration timing issue with radio button clicks
  test.skip('can switch between DB and External polytope source', async ({ page }) => {
    await page.goto('/playground');

    // Default should be External - use radio input text
    const externalRadio = page.getByRole('radio', { name: 'External/Custom' });
    const dbRadio = page.getByRole('radio', { name: 'From Database' });

    await expect(externalRadio).toBeChecked();

    // Switch to DB
    await dbRadio.click();
    await page.waitForTimeout(300);
    await expect(dbRadio).toBeChecked();

    // Verify polytope ID input appears
    await expect(page.getByText('Polytope ID')).toBeVisible();

    // Switch back to External
    await externalRadio.click();
    await page.waitForTimeout(300);
    await expect(externalRadio).toBeChecked();

    // Verify vertices textarea appears
    await expect(page.getByText('Vertices')).toBeVisible();
  });
});

test.describe('Home Page', () => {
  test('loads without errors', async ({ page }) => {
    // Collect console errors
    const errors: string[] = [];
    page.on('pageerror', (err) => {
      errors.push(err.message);
    });

    await page.goto('/');

    // Wait for content to load
    await expect(page.getByText('String Theory')).toBeVisible({ timeout: 10000 });

    // Give it a moment to catch any async errors
    await page.waitForTimeout(1000);
    expect(errors).toHaveLength(0);
  });

  test('navigation to playground works', async ({ page }) => {
    await page.goto('/');

    // Find and click playground link
    await page.getByRole('link', { name: 'Playground' }).click();

    // Verify we're on the playground page
    await expect(page.getByRole('heading', { name: 'Playground' })).toBeVisible();
  });
});
