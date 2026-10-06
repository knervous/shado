import { describe, it, expect } from '@jest/globals';
import { type ShadoConfig, type PendingField } from '../src/decorators';
import { ShaderObject } from '../src/core/ShaderObject';
import { NullEngine } from '@babylonjs/core';

// Simple test class without WASM
class TestStruct extends ShaderObject {
  static readonly shadoConfig: ShadoConfig = { name: 'TestStruct', useWasm: false };
  static readonly shadoFields: readonly PendingField[] = [
    { name: 'fieldA', type: 'u32' },
    { name: 'fieldB', type: 'u32' },
    { name: 'fieldC', type: 'u32' },
  ];
  fieldA!: number;
  fieldB!: number;
  fieldC!: number;

  constructor(engine: any) {
    super(engine);
  }
}

describe('ShaderObject Memory Synchronization', () => {
  it('should sync property writes to arena memory (no WASM)', () => {
    const engine = new NullEngine();
    const obj = new TestStruct(engine);
    
    // Write to properties
    obj.fieldA = 42;
    obj.fieldB = 99999;
    obj.fieldC = 777;

    console.log('=== NO WASM TEST ===');
    console.log('fieldA:', obj.fieldA, '=', (obj as any)._arena.take()[0]);
    console.log('fieldB:', obj.fieldB, '=', (obj as any)._arena.take()[1]);
    console.log('fieldC:', obj.fieldC, '=', (obj as any)._arena.take()[2]);

    // Get the arena data
    const payload = (obj as any).prepareUnifiedForUpload();
    
    // Read directly from payload as if it were header data
    const payloadDV = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    
    // The property values should match the payload DataView values
    expect(payloadDV.getUint32(0, true)).toBe(42);
    expect(payloadDV.getUint32(4, true)).toBe(99999);
    expect(payloadDV.getUint32(8, true)).toBe(777);
    
    // Also check that arena Float32Array sees the values
    const arenaF32 = (obj as any)._arena.take();
    const arenaView = new DataView(arenaF32.buffer, arenaF32.byteOffset, arenaF32.byteLength);
    expect(arenaView.getUint32(0, true)).toBe(42);
    expect(arenaView.getUint32(4, true)).toBe(99999);
    expect(arenaView.getUint32(8, true)).toBe(777);
  });
});
