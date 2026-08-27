#!/usr/bin/env node
// Browser WebGPU scaling benchmark for the receiver-side Shado Lambert loop.
// It uses the runtime's rgba32float address/read pattern and the same light
// attenuation, Lambert floor, and peak compression. This intentionally
// isolates lighting from Babylon scene traversal and rasterization.

import { chromium } from '@playwright/test';
import { createServer } from 'node:http';

function option(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

function integerList(value, label, allowZero = false) {
  return value.split(',').map(entry => {
    const parsed = Number(entry.trim());
    const minimum = allowZero ? 0 : 1;
    if (!Number.isInteger(parsed) || parsed < minimum) {
      throw new Error(`${label} must contain integers >= ${minimum}, got ${entry}`);
    }
    return parsed;
  });
}

const receiverCounts = integerList(option('receivers', '65536,262144'), 'receivers');
const overlaps = integerList(option('overlaps', '0,1,4,8,16,32,64'), 'overlaps', true);
const iterations = integerList(option('iterations', '12'), 'iterations')[0];
const json = process.argv.includes('--json');

const shader = /* wgsl */ `
struct Params {
  sampleCount: u32,
  lightCount: u32,
  textureWidth: u32,
  headersBase: u32,
  indicesBase: u32,
  activeBase: u32,
  minimumLambertBits: u32,
  daylightEnabled: u32,
  dispatchWidth: u32,
  outputCount: u32,
  padding0: u32,
  padding1: u32,
}

@group(0) @binding(0) var fieldTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
@group(0) @binding(2) var<uniform> params: Params;

fn fieldRead(address: u32) -> f32 {
  let texel = address / 4u;
  let packed = textureLoad(
    fieldTexture,
    vec2u(texel % params.textureWidth, texel / params.textureWidth),
    0,
  );
  let lane = address - texel * 4u;
  if (lane == 0u) { return packed.x; }
  if (lane == 1u) { return packed.y; }
  if (lane == 2u) { return packed.z; }
  return packed.w;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
  let sample = globalId.x + globalId.y * params.dispatchWidth * 64u;
  if (sample >= params.sampleCount) { return; }
  let first = u32(fieldRead(params.headersBase));
  let end = first + u32(fieldRead(params.headersBase + 1u));
  let column = f32(sample & 1023u) * 0.00075;
  let row = f32((sample >> 10u) & 1023u) * 0.00075;
  let worldPosition = vec3f(column, 0.0, row);
  let worldNormal = normalize(vec3f(0.15, 1.0, 0.12));
  let minimumLambert = bitcast<f32>(params.minimumLambertBits);
  var lighting = vec3f(0.0);
  for (var reference = first; reference < end; reference = reference + 1u) {
    let light = u32(fieldRead(params.indicesBase + reference));
    if (fieldRead(params.activeBase + light) < 0.5) { continue; }
    let lightRow = light * 8u;
    let position = vec3f(
      fieldRead(lightRow),
      fieldRead(lightRow + 1u),
      fieldRead(lightRow + 2u),
    );
    let range = fieldRead(lightRow + 3u);
    let delta = position - worldPosition;
    let distanceToLight = length(delta);
    let attenuation = max(0.0, 1.0 - distanceToLight / max(range, 0.0001));
    if (attenuation <= 0.0) { continue; }
    let softAttenuation = attenuation * sqrt(attenuation);
    let direction = delta / max(distanceToLight, 0.0001);
    let lambert = max(dot(worldNormal, direction), minimumLambert);
    let radiance = vec3f(
      fieldRead(lightRow + 4u),
      fieldRead(lightRow + 5u),
      fieldRead(lightRow + 6u),
    );
    lighting += radiance * lambert * softAttenuation;
  }
  let peak = max(lighting.r, max(lighting.g, lighting.b));
  if (peak > 0.9) {
    let excess = peak - 0.9;
    let compressedPeak = 0.9 + excess / (1.0 + excess / 0.6);
    lighting *= compressedPeak / peak;
  }
  if (params.daylightEnabled != 0u) {
    // Exact constant-cost daylight/readability tail used by the legacy actor
    // fragment material. The zero-light control quantifies this independently
    // from the variable local-light traversal above.
    let directionalLightColor = vec3f(0.8, 0.78, 0.72);
    let ambientColor = vec3f(0.45, 0.48, 0.55);
    let directionalDaylight = max(
      directionalLightColor.r,
      max(directionalLightColor.g, directionalLightColor.b),
    ) * 1.8;
    let ambientEnergy = max(
      ambientColor.r,
      max(ambientColor.g, ambientColor.b),
    );
    let skyDaylight = (ambientEnergy - 0.22) * 5.0;
    let daylightLevel = clamp(
      max(directionalDaylight, skyDaylight),
      0.0,
      1.0,
    );
    let daylightFloor = mix(
      vec3f(0.045),
      vec3f(0.56, 0.57, 0.6),
      daylightLevel,
    );
    lighting = max(lighting + ambientColor * 0.2, daylightFloor);
    let base = vec3f(0.07, 0.12, 0.18);
    var diffuseSurface = base * lighting;
    let daylightReadability = vec3f(0.065, 0.07, 0.08) * daylightLevel;
    diffuseSurface = max(diffuseSurface, daylightReadability);
    output[sample % params.outputCount] = vec4f(diffuseSurface, 1.0);
  } else {
    output[sample % params.outputCount] = vec4f(lighting, 1.0);
  }
}
`;

const server = createServer((_request, response) => {
  response.writeHead(200, { 'content-type': 'text/html' });
  response.end('<!doctype html><title>Shado light-field WebGPU benchmark</title>');
});
await new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(0, '127.0.0.1', resolve);
});
const address = server.address();
if (!address || typeof address === 'string') {
  throw new Error('Could not create the benchmark HTTP origin');
}

const browser = await chromium.launch({
  headless: true,
  channel: 'chromium',
  args: ['--enable-unsafe-webgpu'],
});

try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${address.port}`);
  const result = await page.evaluate(
    async ({ shader, receiverCounts, overlaps, iterations }) => {
      if (!navigator.gpu) throw new Error('WebGPU is unavailable');
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter is available');
      const timestampQueries = adapter.features.has('timestamp-query');
      const device = await adapter.requestDevice({
        requiredFeatures: timestampQueries ? ['timestamp-query'] : [],
      });
      const module = device.createShaderModule({ code: shader });
      const compilation = await module.getCompilationInfo();
      const errors = compilation.messages.filter(message => message.type === 'error');
      if (errors.length) {
        throw new Error(errors.map(message => message.message).join('\n'));
      }
      const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });

      const percentile = (values, percentileValue) => {
        const sorted = [...values].sort((left, right) => left - right);
        return (
          sorted[
            Math.min(sorted.length - 1, Math.floor((percentileValue / 100) * sorted.length))
          ] ?? 0
        );
      };

      function makeField(lightCount) {
        const headersBase = Math.max(1, lightCount) * 8;
        const indicesBase = headersBase + 2;
        const activeBase = indicesBase + lightCount;
        const arena = new Float32Array(Math.max(4, activeBase + Math.max(1, lightCount)));
        for (let light = 0; light < lightCount; light++) {
          const angle = (light / Math.max(1, lightCount)) * Math.PI * 2;
          const row = light * 8;
          arena[row] = Math.cos(angle) * 3 + 0.4;
          arena[row + 1] = 3.5 + (light % 3) * 0.1;
          arena[row + 2] = Math.sin(angle) * 3 + 0.4;
          arena[row + 3] = 32;
          arena[row + 4] = 0.08;
          arena[row + 5] = 0.042;
          arena[row + 6] = 0.016;
          arena[row + 7] = 0.25;
          arena[indicesBase + light] = light;
          arena[activeBase + light] = 1;
        }
        arena[headersBase] = 0;
        arena[headersBase + 1] = lightCount;
        // Fixed width keeps writeTexture's bytesPerRow 256-byte aligned while
        // retaining the runtime's address-to-texel calculation.
        const textureWidth = 2048;
        const textureHeight = Math.max(1, Math.ceil(arena.length / 4 / textureWidth));
        const upload = new Float32Array(textureWidth * textureHeight * 4);
        upload.set(arena);
        const texture = device.createTexture({
          size: [textureWidth, textureHeight],
          format: 'rgba32float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture(
          { texture },
          upload,
          { bytesPerRow: textureWidth * 16, rowsPerImage: textureHeight },
          [textureWidth, textureHeight]
        );
        return { texture, textureWidth, headersBase, indicesBase, activeBase };
      }

      async function measure(receiverCount, lightCount, daylightEnabled) {
        const field = makeField(lightCount);
        // The output exists only to keep the receiver work externally visible.
        // Bound it so Retina/4K receiver counts do not turn this into a storage
        // capacity benchmark; concurrent writes are acceptable for this sink.
        const outputCount = Math.min(receiverCount, 1_048_576);
        const output = device.createBuffer({
          size: outputCount * 16,
          usage: GPUBufferUsage.STORAGE,
        });
        const totalWorkgroups = Math.ceil(receiverCount / 64);
        const dispatchWidth = Math.min(
          totalWorkgroups,
          device.limits.maxComputeWorkgroupsPerDimension
        );
        const dispatchHeight = Math.ceil(totalWorkgroups / dispatchWidth);
        if (dispatchHeight > device.limits.maxComputeWorkgroupsPerDimension) {
          throw new Error(
            `${receiverCount} receivers exceed this adapter's two-dimensional dispatch capacity`
          );
        }
        const parameterWords = new Uint32Array(12);
        parameterWords[0] = receiverCount;
        parameterWords[1] = lightCount;
        parameterWords[2] = field.textureWidth;
        parameterWords[3] = field.headersBase;
        parameterWords[4] = field.indicesBase;
        parameterWords[5] = field.activeBase;
        parameterWords[6] = new Uint32Array(new Float32Array([0.12]).buffer)[0];
        parameterWords[7] = daylightEnabled ? 1 : 0;
        parameterWords[8] = dispatchWidth;
        parameterWords[9] = outputCount;
        const parameters = device.createBuffer({
          size: parameterWords.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(parameters, 0, parameterWords);
        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: field.texture.createView() },
            { binding: 1, resource: { buffer: output } },
            { binding: 2, resource: { buffer: parameters } },
          ],
        });
        const querySet = timestampQueries
          ? device.createQuerySet({ type: 'timestamp', count: 2 })
          : null;
        const queryResolve = timestampQueries
          ? device.createBuffer({
              size: 16,
              usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
            })
          : null;
        const queryRead = timestampQueries
          ? device.createBuffer({
              size: 16,
              usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            })
          : null;
        const queueSamples = [];
        const gpuSamples = [];
        for (let iteration = 0; iteration < iterations + 4; iteration++) {
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass(
            querySet
              ? {
                  timestampWrites: {
                    querySet,
                    beginningOfPassWriteIndex: 0,
                    endOfPassWriteIndex: 1,
                  },
                }
              : undefined
          );
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(dispatchWidth, dispatchHeight);
          pass.end();
          if (querySet) {
            encoder.resolveQuerySet(querySet, 0, 2, queryResolve, 0);
            encoder.copyBufferToBuffer(queryResolve, 0, queryRead, 0, 16);
          }
          const started = performance.now();
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
          const elapsed = performance.now() - started;
          let gpuMs = null;
          if (queryRead) {
            await queryRead.mapAsync(GPUMapMode.READ);
            const timestamps = new BigUint64Array(queryRead.getMappedRange().slice(0));
            gpuMs = Number(timestamps[1] - timestamps[0]) / 1_000_000;
            queryRead.unmap();
          }
          if (iteration >= 4) {
            queueSamples.push(elapsed);
            if (gpuMs !== null) gpuSamples.push(gpuMs);
          }
        }
        querySet?.destroy();
        queryResolve?.destroy();
        queryRead?.destroy();
        parameters.destroy();
        output.destroy();
        field.texture.destroy();
        return {
          receiverCount,
          lightCount,
          daylightEnabled,
          lightVisits: receiverCount * lightCount,
          queueP50Ms: percentile(queueSamples, 50),
          queueP95Ms: percentile(queueSamples, 95),
          gpuP50Ms: gpuSamples.length ? percentile(gpuSamples, 50) : null,
          gpuP95Ms: gpuSamples.length ? percentile(gpuSamples, 95) : null,
        };
      }

      const rows = [];
      const daylightControls = [];
      for (const receiverCount of receiverCounts) {
        daylightControls.push(await measure(receiverCount, 0, false));
        for (const overlap of overlaps) {
          rows.push(await measure(receiverCount, overlap, true));
        }
      }
      device.destroy();
      return {
        adapter: adapter.info,
        timestampQueries,
        daylightControls,
        rows,
      };
    },
    { shader, receiverCounts, overlaps, iterations }
  );

  const baselines = new Map(
    result.rows
      .filter(row => row.lightCount === 0)
      .map(row => [row.receiverCount, row.gpuP50Ms ?? row.queueP50Ms])
  );
  const daylightControlByReceivers = new Map(
    result.daylightControls.map(row => [row.receiverCount, row.gpuP50Ms ?? row.queueP50Ms])
  );
  const rows = result.rows.map(row => {
    const measured = row.gpuP50Ms ?? row.queueP50Ms;
    const baseline = baselines.get(row.receiverCount) ?? 0;
    const incrementalMs = Math.max(0, measured - baseline);
    return {
      ...row,
      incrementalMs,
      nanosecondsPerVisit:
        row.lightVisits > 0 ? (incrementalMs * 1_000_000) / row.lightVisits : null,
    };
  });

  if (json) {
    console.log(JSON.stringify({ ...result, iterations, rows }, null, 2));
  } else {
    console.log(`\nShado receiver Lambert WebGPU scaling (${iterations} measured iterations)`);
    if (result.adapter?.description) console.log(`adapter: ${result.adapter.description}`);
    console.log(
      `GPU timestamp queries: ${result.timestampQueries ? 'yes' : 'no (queue completion fallback)'}`
    );
    console.log(
      'receivers'.padStart(10),
      'lights/cell'.padStart(12),
      'visits'.padStart(13),
      'GPU p50'.padStart(10),
      'GPU p95'.padStart(10),
      'queue p50'.padStart(11),
      'net/visit'.padStart(12)
    );
    for (const row of rows) {
      console.log(
        row.receiverCount.toLocaleString().padStart(10),
        row.lightCount.toLocaleString().padStart(12),
        row.lightVisits.toLocaleString().padStart(13),
        (row.gpuP50Ms === null ? 'n/a' : `${row.gpuP50Ms.toFixed(3)}ms`).padStart(10),
        (row.gpuP95Ms === null ? 'n/a' : `${row.gpuP95Ms.toFixed(3)}ms`).padStart(10),
        `${row.queueP50Ms.toFixed(3)}ms`.padStart(11),
        (row.nanosecondsPerVisit === null
          ? 'baseline'
          : `${row.nanosecondsPerVisit.toFixed(3)}ns`
        ).padStart(12)
      );
    }
    console.log('\ndaylight/readability fixed cost');
    for (const row of rows.filter(candidate => candidate.lightCount === 0)) {
      const control = daylightControlByReceivers.get(row.receiverCount) ?? 0;
      const measured = row.gpuP50Ms ?? row.queueP50Ms;
      console.log(
        `  ${row.receiverCount.toLocaleString()} receivers: ` +
          `${Math.max(0, measured - control).toFixed(3)}ms GPU p50 delta`
      );
    }
    console.log(
      '\nnet/visit subtracts the zero-light dispatch for the same receiver count. ' +
        'Use receiver count as shaded vertices/fragments, not authored-light count.'
    );
  }
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
