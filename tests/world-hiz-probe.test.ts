import { describe, expect, it } from '@jest/globals';

import { installHeadlessWebGpu } from '../src/devtools/headless-gpu';
import { probeHiZCapability } from '../src/render/hiz-capability-probe';

describe('Hi-Z capability probe (H0.5)', () => {
  it('proves compute -> indirect draw on a real device, and names a limit it lacks', async () => {
    const headless = await installHeadlessWebGpu();
    let device: GPUDevice | undefined;
    try {
      const adapter = await headless.gpu.requestAdapter();
      device = (await adapter!.requestDevice({
        requiredLimits: { maxStorageBufferBindingSize: adapter!.limits.maxStorageBufferBindingSize },
      })) as GPUDevice;
      const report = await probeHiZCapability(device);
      expect(report).toMatchObject({ ok: true, reason: 'ok' });
      expect(report.limits.maxStorageBuffersPerShaderStage).toBeGreaterThanOrEqual(7);

      // A device that cannot bind a 4K pyramid is refused with the reason.
      const small = Object.create(device, {
        limits: { value: { ...device.limits, maxStorageBufferBindingSize: 1 << 20, maxStorageBuffersPerShaderStage: 8, maxComputeWorkgroupSizeX: 256 } },
      }) as GPUDevice;
      const refused = await probeHiZCapability(small);
      expect(refused.ok).toBe(false);
      expect(refused.reason).toMatch(/maxStorageBufferBindingSize/);
    } finally {
      device?.destroy();
      headless.dispose();
    }
  }, 30_000);
});
