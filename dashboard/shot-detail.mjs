// Detailed region crops to eyeball rendering quality at high resolution.
import { chromium } from "playwright";
const URL = process.env.SHOT_URL || "http://localhost:3000";

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width: 1500, height: 1000 },
  deviceScaleFactor: 2,
});
const page = await ctx.newPage();
await page.goto(URL, { waitUntil: "domcontentloaded" });
await page
  .waitForFunction(() => document.body.innerText.includes("OPERATIONAL"), {
    timeout: 30000,
  })
  .catch(() => {});
await page.waitForTimeout(4000);

const h = await page.evaluate(() => document.body.scrollHeight);
console.log("page height", h);

const bands = [
  ["crop-top.png", 0, 520],
  ["crop-charts.png", 520, 560],
  ["crop-stream-ml.png", 1080, 760],
  ["crop-market-cpp.png", 1840, Math.max(100, h - 1840)],
];
for (const [file, y, height] of bands) {
  if (y >= h) continue;
  const clip = { x: 0, y, width: 1500, height: Math.min(height, h - y) };
  await page.screenshot({ path: file, clip, fullPage: true });
  console.log("wrote", file, JSON.stringify(clip));
}
await browser.close();
