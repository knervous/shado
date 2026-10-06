import { NullEngine } from '@babylonjs/core';
import { type ShadoConfig, type PendingField } from '../src/decorators';
import { ShadoActor } from '../src/extensions/ShadoActor';

class SpecializedActor extends ShadoActor {
  static readonly shadoConfig: ShadoConfig = { name: 'SpecializedActor', useWasm: false };
  static readonly shadoFields: readonly PendingField[] = [
    { name: 'entityId', type: 'u32' },
  ];
  entityId!: number;
}

describe('Shado schema inheritance', () => {
  it('builds a subclass schema after the base class has been initialized', async () => {
    const engine = new NullEngine();
    await ShadoActor.initialize(engine, { wasm: false });

    const base = ShadoActor.getSchema();
    const specialized = SpecializedActor.getSchema();

    expect(base.name).toBe('ShadoActor');
    expect(base.fields.some(field => field.name === 'entityId')).toBe(false);
    expect(specialized.name).toBe('SpecializedActor');
    expect(specialized.fields.some(field => field.name === 'translation')).toBe(true);
    expect(specialized.fields.some(field => field.name === 'entityId')).toBe(true);

    engine.dispose();
  });
});
