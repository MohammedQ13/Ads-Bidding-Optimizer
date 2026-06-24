// Captures the live console in motion and encodes a GIF (no ffmpeg needed).
// Usage: SHOT_URL=https://...vercel.app node make-gif.mjs
import { chromium } from "playwright";
import gifPkg from "gif-encoder-2";
import pngPkg from "pngjs";

const GIFEncoder = gifPkg.default ?? gifPkg;
const { PNG } = pngPkg;

const URL = process.env.SHOT_URL || "http://localhost:3000";
const W = 1280, H = 720, FRAMES = 16, SPACING_MS = 1100, OUT = "demo.gif";

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: W, height: 860 }, deviceScaleFactor: 1 });
const page = await ctx.newPage();
await page.goto(URL, { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => document.body.innerText.includes("OPERATIONAL"), { timeout: 30000 }).catch(() => {});
await page.waitForTimeout(3000);

const clip = { x: 0, y: 0, width: W, height: H };
const frames = [];
for (let i = 0; i < FRAMES; i++) {
  const buf = await page.screenshot({ clip });
  frames.push(PNG.sync.read(buf).data);
  console.log(`frame ${i + 1}/${FRAMES}`);
  if (i < FRAMES - 1) await page.waitForTimeout(SPACING_MS);
}
await browser.close();

const enc = new GIFEncoder(W, H, "neuquant", true);
enc.setDelay(650);
enc.setRepeat(0);
enc.setQuality(10);
enc.start();
// gif-encoder-2 reads pixels via ctx.getImageData(...).data — give it a shim
for (const f of frames) enc.addFrame({ getImageData: () => ({ data: f }) });
enc.finish();

const { writeFileSync } = await import("fs");
writeFileSync(OUT, enc.out.getData());
console.log(`wrote ${OUT}`);
