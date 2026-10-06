/* Shado GPU struct layouts, declared as explicit static tables.
 *
 * A Shado class describes its packed layout with two statics instead of
 * member decorators (which compile nowhere outside a decorator runtime):
 *
 *   class MyActor extends ShadoActor {
 *     static readonly shadoConfig: ShadoConfig = { name: 'MyActor' };
 *     static readonly shadoFields: readonly PendingField[] = [
 *       { name: 'tint', type: 'vec4' },
 *     ];
 *     tint!: Float32Array;
 *   }
 *
 * `shadoFields` lists only the class's OWN fields; readFields concatenates
 * the chain base-first, as the former @field decorators accumulated them.
 * `shadoConfig` inherits like any static: the nearest declaration wins. */

export type PendingField = { name: string; type: any };

export type ShadoConfig = {
  name?: string;
  useWasm?: boolean;
  /** GPU is the existing float/storage layout; net uses the fixed byte-addressed ABI emitter. */
  layout?: 'gpu' | 'net';
  schemaId?: number;
  version?: number;
};

type ShadoClass = { shadoConfig?: ShadoConfig; shadoFields?: readonly PendingField[] };

export function readClassMeta(ctor: any): ShadoConfig {
  return ((ctor as ShadoClass | undefined)?.shadoConfig ?? {}) as ShadoConfig;
}

export function readFields(ctor: any): PendingField[] {
  const chain: ShadoClass[] = [];
  for (let c = ctor; c && c !== Function.prototype; c = Object.getPrototypeOf(c)) chain.push(c);
  const fields: PendingField[] = [];
  for (const c of chain.reverse()) {
    if (Object.prototype.hasOwnProperty.call(c, 'shadoFields')) fields.push(...(c.shadoFields ?? []));
  }
  return fields;
}
