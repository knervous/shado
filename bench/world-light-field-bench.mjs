#!/usr/bin/env node
// CPU construction, mutation, and memory scaling for the Shado world-light
// field. GPU receiver-loop cost is measured separately by
// world-light-field-webgpu-bench.mjs.

import { NullEngine, Scene } from '@babylonjs/core';
import { ShadoWorldLightBuffer } from '../dist/index.js';

function option(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer, got ${value}`);
  }
  return parsed;
}

function percentile(samples, percentileValue) {
  const sorted = [...samples].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.floor((percentileValue / 100) * sorted.length));
  return sorted[index] ?? 0;
}

function sample(operation, warmups, iterations) {
  const values = [];
  for (let iteration = 0; iteration < warmups + iterations; iteration++) {
    const started = performance.now();
    operation(iteration);
    const elapsed = performance.now() - started;
    if (iteration >= warmups) values.push(elapsed);
  }
  return {
    p50: percentile(values, 50),
    p95: percentile(values, 95),
  };
}

function makeLight(index, position) {
  return {
    id: `bench-light-${index}`,
    name: `Benchmark light ${index}`,
    source: 'standalone',
    enabled: true,
    position,
    color: [1, 0.52, 0.2],
    intensity: 13,
    range: 14,
    radius: 0.25,
    castsShadows: false,
    bake: false,
    runtime: true,
    cellId: -1,
    visibilityRegion: -1,
    phaseMask: index % 2 === 0 ? 1 : 2,
    tags: ['benchmark'],
    metadata: { kind: 'torch-flame' },
    activation: {
      mode: 'night',
      onHour: 18,
      offHour: 6,
      transitionMinutes: 25,
    },
    flicker: {
      profile: 'flame',
      amplitude: 0.065,
      speed: 6.5,
    },
  };
}

function makeWorld(lightCount, layout) {
  const gridSide = Math.ceil(Math.sqrt(lightCount));
  const worldSize = Math.max(64, gridSide * 32);
  const center = worldSize * 0.5;
  const pointLights = Array.from({ length: lightCount }, (_, index) => {
    const position =
      layout === 'dense'
        ? [center + (index % 3) * 0.1, 4, center + (index % 5) * 0.1]
        : [(index % gridSide) * 32 + 16, 4, Math.floor(index / gridSide) * 32 + 16];
    return makeLight(index, position);
  });
  return {
    name: `world-light-field-${layout}-${lightCount}`,
    bounds: { min: [0, 0, 0], max: [worldSize, 32, worldSize] },
    tiles: { size: 32 },
    pointLights,
  };
}

const lightCounts = option('lights', '64,256,1024,4096')
  .split(',')
  .map(value => positiveInteger(value.trim(), 'light count'));
const layouts = option('layouts', 'sparse,dense')
  .split(',')
  .map(value => value.trim())
  .filter(value => value === 'sparse' || value === 'dense');
const iterations = positiveInteger(option('iterations', '12'), 'iterations');
const buildIterations = positiveInteger(option('build-iterations', '4'), 'build iterations');
const json = process.argv.includes('--json');

if (layouts.length === 0) {
  throw new Error('--layouts must contain sparse and/or dense');
}

const engine = new NullEngine({ renderWidth: 16, renderHeight: 16 });
const scene = new Scene(engine);
const rows = [];

for (const layout of layouts) {
  for (const lightCount of lightCounts) {
    const world = makeWorld(lightCount, layout);
    let latestField;
    const buildSamples = [];
    for (let iteration = 0; iteration < buildIterations + 1; iteration++) {
      const started = performance.now();
      const field = new ShadoWorldLightBuffer(scene, world, {}, { cellSize: 32 });
      const elapsed = performance.now() - started;
      if (iteration > 0) buildSamples.push(elapsed);
      latestField?.dispose();
      latestField = field;
    }

    const field = latestField;
    let phaseMask = 1;
    const reduction = sample(
      () => {
        phaseMask = phaseMask === 1 ? 2 : 1;
        field.reduce([], {}, [0, 0, 0], { activePhaseMask: phaseMask });
      },
      2,
      iterations
    );

    const cachedReduction = sample(
      () => {
        field.reduce([], {}, [0, 0, 0], { activePhaseMask: phaseMask });
      },
      2,
      iterations
    );

    const runtimeTick = sample(
      iteration => field.tickRuntime(80, 22 + (iteration % 10) * 0.001),
      2,
      iterations
    );

    const nonStructural = sample(
      iteration => {
        const index = iteration % lightCount;
        field.updateLight(index, { intensity: 12.5 + (iteration % 2) });
      },
      2,
      iterations
    );

    const structural = sample(
      iteration => {
        const index = iteration % lightCount;
        const source = world.pointLights[index].position;
        field.updateLight(index, {
          position: [source[0] + (iteration % 2 ? 0.25 : -0.25), source[1], source[2]],
        });
      },
      1,
      Math.min(iterations, 6)
    );

    rows.push({
      layout,
      lightCount,
      maxLightsPerCell: field.diagnostics.maxLightsPerCell,
      lightReferences: field.diagnostics.lightReferences,
      arenaBytes: field.diagnostics.arenaBytes,
      bytesPerLight: field.diagnostics.arenaBytes / lightCount,
      build: {
        p50: percentile(buildSamples, 50),
        p95: percentile(buildSamples, 95),
      },
      reduction,
      cachedReduction,
      runtimeTick,
      nonStructural,
      structural,
    });
    field.dispose();
  }
}

scene.dispose();
engine.dispose();

if (json) {
  console.log(JSON.stringify({ iterations, buildIterations, rows }, null, 2));
} else {
  console.log(`\nShado world-light field CPU scaling (${iterations} measured iterations)`);
  console.log(
    'layout'.padEnd(8),
    'lights'.padStart(7),
    'max/cell'.padStart(9),
    'refs'.padStart(10),
    'arena MiB'.padStart(11),
    'build p50'.padStart(11),
    'tick p50'.padStart(10),
    'phase p50'.padStart(11),
    'cached'.padStart(9),
    'edit p50'.padStart(10),
    'repack p50'.padStart(12)
  );
  for (const row of rows) {
    console.log(
      row.layout.padEnd(8),
      row.lightCount.toLocaleString().padStart(7),
      row.maxLightsPerCell.toLocaleString().padStart(9),
      row.lightReferences.toLocaleString().padStart(10),
      (row.arenaBytes / (1024 * 1024)).toFixed(3).padStart(11),
      `${row.build.p50.toFixed(3)}ms`.padStart(11),
      `${row.runtimeTick.p50.toFixed(3)}ms`.padStart(10),
      `${row.reduction.p50.toFixed(3)}ms`.padStart(11),
      `${row.cachedReduction.p50.toFixed(3)}ms`.padStart(9),
      `${row.nonStructural.p50.toFixed(3)}ms`.padStart(10),
      `${row.structural.p50.toFixed(3)}ms`.padStart(12)
    );
  }
  console.log(
    '\nTick/edit timings include the current full compact-texture upload. ' +
      'Repack includes cell adjacency rebuild and texture recreation.'
  );
}
