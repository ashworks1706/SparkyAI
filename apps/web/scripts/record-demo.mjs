// Renders the Discord demo frame by frame and encodes it to public/demo with ffmpeg.
// Usage: CHROMIUM=/path/to/chrome node scripts/record-demo.mjs [ask|setup]
import { spawn } from "node:child_process";
import { rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const out = `${root}public/demo`;
const FPS = 30;
const VIDEOS = { ask: { file: "sparky-demo", length: 18 }, setup: { file: "sparky-setup", length: 30 } };
const name = process.argv[2] ?? "ask";
const { file, length: LENGTH } = VIDEOS[name];
const SIZE = { width: 1600, height: 1000 };

const server = await createServer({ root, server: { host: "127.0.0.1", port: 5181 }, logLevel: "error" });
await server.listen();
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM });
const page = await browser.newPage({ viewport: SIZE });
await page.goto(`http://127.0.0.1:5181/demo.html?video=${name}`, { waitUntil: "networkidle" });
await page.waitForFunction(() => typeof window.seekDemo === "function");
await page.evaluate(() => document.fonts.ready);

const encode = (args) => spawn("ffmpeg", ["-y", "-loglevel", "error", ...args], { stdio: ["pipe", "inherit", "inherit"] });
const video = encode([
  "-f", "image2pipe", "-framerate", String(FPS), "-i", "-",
  "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
  `${out}/${file}.mp4`,
]);
for (let i = 0; i < FPS * LENGTH; i++) {
  await page.evaluate((t) => window.seekDemo(t), i / FPS);
  video.stdin.write(await page.screenshot({ type: "png" }));
}
video.stdin.end();
await new Promise((done) => video.on("close", done));

await page.evaluate((t) => window.seekDemo(t), LENGTH);
await page.screenshot({ path: `${out}/${file}.png` });
const poster = encode(["-i", `${out}/${file}.png`, "-q:v", "80", `${out}/${file}.webp`]);
await new Promise((done) => poster.on("close", done));
await rm(`${out}/${file}.png`);

await browser.close();
await server.close();
