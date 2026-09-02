import { describe, expect, it } from '@jest/globals';

import { installHeadlessWebGpu, TEXTURE_USAGE } from '../src/devtools/headless-gpu';

const PRIMITIVE_INDEX_SHADER = /* wgsl */ `
enable primitive_index;

@fragment
fn fs(@builtin(primitive_index) primitiveIndex: u32) -> @location(0) vec4f {
  return vec4f(f32(primitiveIndex));
}
`;

describe('headless webgpu backend', () => {
  it('supports primitive-index, texture views, and native WGSL language reporting', async () => {
    const headless = await installHeadlessWebGpu();
    let device: any;

    try {
      const adapter = await headless.gpu.requestAdapter();
      expect(adapter).not.toBeNull();
      expect(adapter.features.has('primitive-index')).toBe(true);
      expect(adapter.features.has('texture-component-swizzle')).toBe(true);
      expect(headless.gpu.wgslLanguageFeatures.size).toBeGreaterThan(0);
      expect(headless.gpu.wgslLanguageFeatures.has('uniform_buffer_standard_layout')).toBe(true);

      device = await adapter.requestDevice({ requiredFeatures: ['primitive-index'] });
      expect(device.features.has('primitive-index')).toBe(true);

      const texture = device.createTexture({
        size: [4, 4],
        format: 'rgba8unorm',
        usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.RENDER_ATTACHMENT,
      });
      expect(texture.createView()).toBeDefined();
      texture.destroy();

      const shader = device.createShaderModule({ code: PRIMITIVE_INDEX_SHADER });
      const compilation = await shader.getCompilationInfo();
      expect(compilation.messages).toHaveLength(0);
    } finally {
      device?.destroy?.();
      headless.dispose();
    }
  });
});
