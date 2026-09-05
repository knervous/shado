#!/usr/bin/env node

import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import path from 'node:path';

function option(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

function numericOption(name, fallback) {
  const value = Number(option(name, fallback));
  return Number.isFinite(value) ? value : fallback;
}

async function waitForOrigin(url) {
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Vite is still starting.
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out waiting for ${url}`);
}

const counts = option('counts', '10000,100000')
  .split(',')
  .map(Number)
  .filter(value => Number.isFinite(value) && value > 0);
const scenarios = option('scenarios', 'field,dense')
  .split(',')
  .filter(value => value === 'field' || value === 'dense');
const backend = option('backend', 'webgl2');
const gpuMotion = process.argv.includes('--gpu-motion');
const warmupFrames = numericOption('warmup-frames', 20);
const sampleFrames = numericOption('sample-frames', 90);
const port = numericOption('port', 4184);
const origin = `http://127.0.0.1:${port}`;
const sandboxDirectory = path.resolve(import.meta.dirname, '../sandbox');
const vite = spawn(
  'npm',
  ['run', 'start', '--', '--host', '127.0.0.1', '--port', String(port)],
  { cwd: sandboxDirectory, stdio: ['ignore', 'ignore', 'inherit'] }
);
const browser = await chromium.launch({
  headless: true,
  channel: 'chromium',
  args: backend === 'webgpu' ? ['--enable-unsafe-webgpu'] : [],
});

try {
  await waitForOrigin(`${origin}/sprites-2d`);
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.on('console', message => {
    if (message.type() === 'error') console.error(`[browser] ${message.text()}`);
  });
  const rows = [];
  const renderPaths = gpuMotion && backend === 'webgpu'
    ? ['optimized-cull', 'optimized-all', 'full']
    : ['optimized', 'full'];
  for (const scenario of scenarios) {
    for (const count of counts) {
      for (const renderPath of renderPaths) {
        const query = new URLSearchParams({
          renderer: 'babylonjs',
          backend,
          loadTest: '1',
          scenario,
          count: String(count),
          warmupFrames: String(warmupFrames),
          sampleFrames: String(sampleFrames),
        });
        if (renderPath === 'full') query.set('path', 'legacy');
        if (gpuMotion && renderPath !== 'full') query.set('gpuMotion', '1');
        if (renderPath === 'optimized-cull') query.set('gpuCulling', '1');
        if (renderPath === 'optimized-all') query.set('gpuCulling', '0');
        await page.goto(`${origin}/sprites-2d?${query}`, { waitUntil: 'domcontentloaded' });
        const result = await page.evaluate(async () => {
          const deadline = performance.now() + 120_000;
          while (!(window).shadoSprite2DLoadTest) {
            if (performance.now() >= deadline) throw new Error('Load test did not initialize');
            await new Promise(resolve => setTimeout(resolve, 25));
          }
          return await (window).shadoSprite2DLoadTest.ready;
        });
        rows.push(result);
        console.log(
          `${scenario.padEnd(5)} ${count.toLocaleString().padStart(8)} ${renderPath.padEnd(9)} ` +
          `${result.submittedSprites.toLocaleString().padStart(8)} submitted  ` +
          `${result.setupMs.toFixed(1).padStart(7)} ms setup  ` +
          `${result.frameMs.p50.toFixed(2).padStart(7)} ms p50  ` +
          `${result.frameMs.p95.toFixed(2).padStart(7)} ms p95  ` +
          `${result.gpuMs ? result.gpuMs.p50.toFixed(2).padStart(7) : '    n/a'} ms GPU  ` +
          `${result.effectiveFps.toFixed(1).padStart(6)} fps  ` +
          `${(result.workingSetBytes / 1048576).toFixed(2).padStart(6)} MiB  ` +
          `${result.gpuCulling ? result.indirectDraw ? 'GPU-cull+indirect' : 'GPU-cull' : 'CPU/static-cull'}`
        );
      }
    }
  }

  console.log('\nComparisons:');
  for (let index = 0; index < rows.length; index += renderPaths.length) {
    const group = rows.slice(index, index + renderPaths.length);
    const optimized = group[0];
    const full = group.at(-1);
    if (!optimized || !full) continue;
    const faster = full.frameMs.p50 / Math.max(0.0001, optimized.frameMs.p50);
    const saved = 100 * (1 - optimized.workingSetBytes / full.workingSetBytes);
    const allGpu = group.find(row => row.path === 'optimized' && !row.gpuCulling);
    const cullingGain = allGpu
      ? `, ${(allGpu.frameMs.p50 / Math.max(0.0001, optimized.frameMs.p50)).toFixed(2)}x CPU-frame cull gain, ` +
        `${((allGpu.gpuMs?.p50 ?? 0) / Math.max(0.0001, optimized.gpuMs?.p50 ?? 0)).toFixed(2)}x GPU-time cull gain`
      : '';
    console.log(
      `${optimized.scenario.padEnd(5)} ${optimized.sprites.toLocaleString().padStart(8)}: ` +
      `${faster.toFixed(2)}x p50 speedup (full/optimized), ` +
      `${saved.toFixed(1)}% record memory saved${cullingGain}`
    );
  }
  console.log(`\nSHADO_SPRITE_2D_LOAD_RESULT=${JSON.stringify({ backend, rows })}`);
} finally {
  await browser.close();
  vite.kill('SIGTERM');
}
