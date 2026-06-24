// Screenshot helper for the live console. Usage: node shot.mjs
import { chromium } from "playwright";

const URL = process.env.SHOT_URL || "http://localhost:3000";

async function shoot(page, theme, file) {
  await page.addInitScript((t) => {
    try {
      localStorage.setItem("theme", t);
    } catch {}
  }, theme);
  await page.goto(URL, { waitUntil: "domcontentloaded" });
  // wait until live data has populated, then let the charts build some history
  await page
    .waitForFunction(() => document.body.innerText.includes("OPERATIONAL"), {
      timeout: 30000,
    })
    .catch(() => {});
  await page.waitForTimeout(7000);
  await page.screenshot({ path: file, fullPage: true });
  console.log("wrote", file);
}

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width: 1600, height: 1200 },
  deviceScaleFactor: 1.5,
});
const page = await ctx.newPage();

await shoot(page, "dark", "shot-console-dark.png");
await shoot(page, "light", "shot-console-light.png");

await browser.close();
console.log("done");
