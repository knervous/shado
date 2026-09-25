import { AOS_SCALAR_INFO, alignUp, type AoSScalarType } from '../schema/AoSLayout';

/**
 * A closed set of names stored as its index: `u8`, or `u16` past 256 values.
 * Readers see the name; an index outside the set reads as the first name.
 */
export interface NetEnumType {
  readonly enum: readonly string[];
}

/**
 * `true`: the field may be absent, recorded in a presence bit, and reads back
 * as `undefined` (Rust `Option`). `'null'`: the same, but absent reads back as
 * `null`, for values the other side distinguishes with `null`.
 */
export type NetOptional = true | 'null';

export interface NetScalarFieldSpec {
  readonly id: number;
  readonly name: string;
  readonly type: AoSScalarType | NetEnumType;
  readonly count?: number;
  readonly visibility?: 'public' | 'private';
  readonly optional?: NetOptional;
}

export interface NetStructRef {
  readonly struct: string;
  /** Inline only these fields from the referenced struct. Omit to inline all fields. */
  readonly pick?: readonly string[];
  /** Inline every non-private field. Mutually exclusive with pick. */
  readonly visibility?: 'public';
}

export interface NetStructFieldSpec {
  readonly id: number;
  readonly name: string;
  readonly type: NetStructRef;
  readonly count?: number;
  readonly visibility?: 'public' | 'private';
  readonly optional?: NetOptional;
}

/**
 * A list stored out of line in the packet's heap: scalars, or fixed records
 * (which may themselves carry variable-length fields).
 */
export interface NetListType {
  readonly list: AoSScalarType | 'str' | NetStructRef;
}

/** Variable-length data: UTF-8 text, raw bytes, or a list. */
export type NetVarType = 'str' | 'bytes' | NetListType;

/**
 * A field whose data lives in the heap after the records. The record keeps an
 * 8-byte slot, `u32 heap offset | u32 element count`, so the fixed region never
 * moves and a reader can reach any value without walking anything before it.
 */
export interface NetVarFieldSpec {
  readonly id: number;
  readonly name: string;
  readonly type: NetVarType;
  readonly visibility?: 'public' | 'private';
  readonly optional?: NetOptional;
}

export type NetFieldSpec = NetScalarFieldSpec | NetStructFieldSpec | NetVarFieldSpec;

export interface NetStructSpec {
  readonly name: string;
  readonly layout: 'net';
  /** Packet identity. Omit for a record that only ever travels inside another. */
  readonly schemaId?: number;
  readonly version?: number;
  readonly storage?: 'aos' | 'soa';
  readonly variants?: readonly NetVariantSpec[];
  readonly fields: readonly NetFieldSpec[];
}

export interface NetVariantSpec {
  readonly name: string;
  readonly tag: number;
  readonly fields: readonly string[];
}

export interface NetScalarFieldLayout extends Omit<NetScalarFieldSpec, 'type'> {
  readonly kind: 'scalar';
  /** Storage type; an enum field stores its index as `u8` or `u16`. */
  readonly type: AoSScalarType;
  /** The names an enum field's index selects from. */
  readonly enumValues?: readonly string[];
  readonly count: number;
  readonly byteOffset: number;
  readonly byteSize: number;
  readonly alignment: number;
  /** Bit in the record's presence words, for an optional field. */
  readonly presenceBit?: number;
}

export interface NetStructFieldLayout extends NetStructFieldSpec {
  readonly kind: 'struct';
  readonly count: number;
  readonly byteOffset: number;
  readonly byteSize: number;
  readonly alignment: number;
  readonly struct: NetStructLayout;
  readonly presenceBit?: number;
}

export interface NetVarFieldLayout extends NetVarFieldSpec {
  readonly kind: 'var';
  readonly presenceBit?: number;
  /** `strList`: a list of strings, each element an 8-byte slot of its own. */
  readonly varKind: 'str' | 'bytes' | 'scalarList' | 'strList' | 'structList';
  /** Element scalar type for `str`/`bytes` (u8) and scalar lists. */
  readonly element?: AoSScalarType;
  /** Element record for struct lists. */
  readonly struct?: NetStructLayout;
  readonly elementSize: number;
  readonly elementAlignment: number;
  readonly count: 1;
  readonly byteOffset: number;
  readonly byteSize: 8;
  readonly alignment: 4;
}

export type NetFieldLayout = NetScalarFieldLayout | NetStructFieldLayout | NetVarFieldLayout;

/** Bytes of the in-record slot a variable-length field occupies. */
export const NET_VAR_SLOT_BYTES = 8;

export function isNetVarType(type: NetFieldSpec['type']): type is NetVarType {
  return type === 'str' || type === 'bytes' || (typeof type === 'object' && 'list' in type);
}

export interface NetStructLayout {
  readonly name: string;
  readonly layout: 'net';
  readonly schemaId: number;
  readonly version: number;
  readonly byteSize: number;
  readonly stride: number;
  readonly alignment: number;
  readonly schemaHash: bigint;
  readonly fields: readonly NetFieldLayout[];
  /** Projection layouts are inline-only and do not have their own packet identity. */
  readonly projectionOf?: string;
  readonly storage: 'aos' | 'soa';
  readonly variants: readonly NetVariantSpec[];
  /** True when this record, or any record inside it, has heap data. */
  readonly variable: boolean;
  /**
   * `u32` words at the start of the record holding one presence bit per
   * optional field, in field order. Zero when no field is optional.
   */
  readonly presenceWords: number;
}

/**
 * Compile the fixed-width network ABI. Struct references are inline regions, so a
 * packet remains one contiguous block suitable for Wasm memory and GPU upload.
 */
export function compileNetLayouts(
  specs: readonly NetStructSpec[]
): ReadonlyMap<string, NetStructLayout> {
  const byName = new Map<string, NetStructSpec>();
  const schemaIds = new Set<number>();
  for (const spec of specs) {
    if (!/^[$A-Z_a-z][$\w]*$/.test(spec.name) || byName.has(spec.name)) {
      throw new Error(`Invalid or duplicate net struct name: ${spec.name}`);
    }
    if (spec.layout !== 'net') throw new Error(`${spec.name} must use layout \"net\"`);
    if (
      spec.schemaId !== undefined &&
      (!Number.isSafeInteger(spec.schemaId) ||
        spec.schemaId <= 0 ||
        spec.schemaId > 0xffffffff ||
        schemaIds.has(spec.schemaId))
    ) {
      throw new Error(`Invalid or duplicate net schema ID: ${spec.schemaId}`);
    }
    byName.set(spec.name, spec);
    if (spec.schemaId !== undefined) schemaIds.add(spec.schemaId);
  }

  const compiled = new Map<string, NetStructLayout>();
  const compiling = new Set<string>();

  const compile = (name: string): NetStructLayout => {
    const existing = compiled.get(name);
    if (existing) return existing;
    const spec = byName.get(name);
    if (!spec) throw new Error(`Unknown net struct reference: ${name}`);
    if (compiling.has(name)) throw new Error(`Recursive net struct reference involving ${name}`);
    compiling.add(name);
    const result = compileFields(spec, spec.fields, compile);
    compiling.delete(name);
    compiled.set(name, result);
    return result;
  };

  for (const spec of specs) compile(spec.name);
  return compiled;
}

function compileFields(
  owner: NetStructSpec,
  fields: readonly NetFieldSpec[],
  resolve: (name: string) => NetStructLayout,
  projectionOf?: string
): NetStructLayout {
  const names = new Set<string>();
  const ids = new Set<number>();
  const laidOut: NetFieldLayout[] = [];
  const optionalCount = fields.filter(field => field.optional).length;
  const presenceWords = Math.ceil(optionalCount / 32);
  let cursor = presenceWords * 4;
  let alignment = presenceWords ? 4 : 1;
  let nextPresenceBit = 0;

  for (const field of fields) {
    if (!field.name || names.has(field.name))
      throw new Error(`Duplicate or empty field name: ${field.name}`);
    if (!Number.isSafeInteger(field.id) || field.id <= 0 || ids.has(field.id)) {
      throw new Error(`Invalid or duplicate field id: ${field.id}`);
    }
    const count = ('count' in field ? field.count : undefined) ?? 1;
    if (!Number.isSafeInteger(count) || count <= 0) {
      throw new Error(`${owner.name}.${field.name} count must be a positive safe integer`);
    }
    names.add(field.name);
    ids.add(field.id);
    if (field.optional && field.optional !== true && field.optional !== 'null') {
      throw new Error(`${owner.name}.${field.name}: optional must be true or 'null'`);
    }
    const presence = field.optional ? { presenceBit: nextPresenceBit++ } : {};

    if (isNetVarType(field.type)) {
      if (count !== 1)
        throw new Error(
          `${owner.name}.${field.name}: variable fields cannot be arrays; use a list`
        );
      const type = field.type;
      let varLayout: Omit<
        NetVarFieldLayout,
        keyof NetVarFieldSpec | 'kind' | 'count' | 'byteOffset' | 'byteSize' | 'alignment'
      >;
      if (type === 'str' || type === 'bytes') {
        varLayout = { varKind: type, element: 'u8', elementSize: 1, elementAlignment: 1 };
      } else if (type.list === 'str') {
        varLayout = {
          varKind: 'strList',
          elementSize: NET_VAR_SLOT_BYTES,
          elementAlignment: 4,
        };
      } else if (typeof type.list === 'string') {
        const info = AOS_SCALAR_INFO[type.list];
        if (!info) throw new Error(`Unsupported net list element type: ${String(type.list)}`);
        varLayout = {
          varKind: 'scalarList',
          element: type.list,
          elementSize: info.byteSize,
          elementAlignment: info.alignment,
        };
      } else {
        const element = resolveStructRef(owner, field.name, type.list, resolve);
        varLayout = {
          varKind: 'structList',
          struct: element,
          elementSize: element.stride,
          elementAlignment: element.alignment,
        };
      }
      cursor = alignUp(cursor, 4);
      laidOut.push(
        Object.freeze({
          ...(field as NetVarFieldSpec),
          ...varLayout,
          ...presence,
          kind: 'var',
          count: 1,
          byteOffset: cursor,
          byteSize: NET_VAR_SLOT_BYTES,
          alignment: 4,
        } as NetVarFieldLayout)
      );
      cursor += NET_VAR_SLOT_BYTES;
      alignment = Math.max(alignment, 4);
      continue;
    }

    const enumValues =
      typeof field.type === 'object' && 'enum' in field.type ? field.type.enum : undefined;
    if (typeof field.type === 'string' || enumValues) {
      if (enumValues) {
        if (count !== 1)
          throw new Error(`${owner.name}.${field.name}: enum fields cannot be arrays`);
        if (!enumValues.length || new Set(enumValues).size !== enumValues.length) {
          throw new Error(`${owner.name}.${field.name}: enum values must be unique and non-empty`);
        }
        if (enumValues.length > 0xffff) {
          throw new Error(`${owner.name}.${field.name}: enum has more than 65536 values`);
        }
      }
      const type: AoSScalarType = enumValues
        ? enumValues.length > 0x100
          ? 'u16'
          : 'u8'
        : (field.type as AoSScalarType);
      const info = AOS_SCALAR_INFO[type];
      if (!info) throw new Error(`Unsupported net scalar type: ${String(field.type)}`);
      cursor = alignUp(cursor, info.alignment);
      const byteSize = info.byteSize * count;
      laidOut.push(
        Object.freeze({
          ...(field as Omit<NetScalarFieldSpec, 'type'>),
          type,
          ...(enumValues ? { enumValues: Object.freeze([...enumValues]) } : {}),
          ...presence,
          kind: 'scalar',
          count,
          byteOffset: cursor,
          byteSize,
          alignment: info.alignment,
        })
      );
      cursor += byteSize;
      alignment = Math.max(alignment, info.alignment);
      continue;
    }

    const structField = field as NetStructFieldSpec;
    const nested = resolveStructRef(owner, field.name, structField.type, resolve);
    cursor = alignUp(cursor, nested.alignment);
    const byteSize = nested.stride * count;
    laidOut.push(
      Object.freeze({
        ...structField,
        ...presence,
        kind: 'struct',
        count,
        byteOffset: cursor,
        byteSize,
        alignment: nested.alignment,
        struct: nested,
      })
    );
    cursor += byteSize;
    alignment = Math.max(alignment, nested.alignment);
  }

  const stride = alignUp(cursor, alignment);
  const version = owner.version ?? 1;
  validateVariants(owner, laidOut);
  const storage = owner.storage ?? 'aos';
  const variants = owner.variants ?? [];
  const variable = laidOut.some(
    field => field.kind === 'var' || (field.kind === 'struct' && field.struct.variable)
  );
  if (variable && storage === 'soa') {
    throw new Error(`${owner.name}: SoA packets cannot carry variable-length fields`);
  }
  if (presenceWords && storage === 'soa') {
    throw new Error(`${owner.name}: SoA packets cannot carry optional fields`);
  }
  const extra = (field: NetFieldLayout) =>
    `${field.presenceBit !== undefined ? `?${field.presenceBit}${field.optional === 'null' ? 'n' : ''}` : ''}${
      field.kind === 'scalar' && field.enumValues ? `=${field.enumValues.join(',')}` : ''
    }`;
  const normalized = `${version}|${storage}|${stride}|${laidOut
    .map(field =>
      field.kind === 'scalar'
        ? `${field.id}:${field.type}:${field.count}:${field.byteOffset}:${field.visibility ?? 'public'}${extra(field)}`
        : field.kind === 'var'
          ? `${field.id}:var:${field.varKind}:${field.struct ? field.struct.schemaHash.toString(16) : field.element}:${field.byteOffset}:${field.visibility ?? 'public'}${extra(field)}`
          : `${field.id}:struct:${field.count}:${field.byteOffset}:${field.struct.schemaHash.toString(16)}:${field.visibility ?? 'public'}${extra(field)}`
    )
    .join('|')}|${variants.map(variant => `${variant.tag}:${variant.fields.join(',')}`).join(';')}`;
  return Object.freeze({
    name: owner.name,
    layout: 'net',
    schemaId: owner.schemaId ?? 0,
    version,
    byteSize: cursor,
    stride,
    alignment,
    schemaHash: fnv1a64(normalized),
    fields: Object.freeze(laidOut),
    ...(projectionOf ? { projectionOf } : {}),
    storage,
    variable,
    presenceWords,
    variants: Object.freeze(
      variants.map(variant =>
        Object.freeze({ ...variant, fields: Object.freeze([...variant.fields]) })
      )
    ),
  });
}

function resolveStructRef(
  owner: NetStructSpec,
  fieldName: string,
  ref: NetStructRef,
  resolve: (name: string) => NetStructLayout
): NetStructLayout {
  const referenced = resolve(ref.struct);
  let nested = referenced;
  if (ref.pick && ref.visibility) {
    throw new Error(`${owner.name}.${fieldName} cannot specify both pick and visibility`);
  }
  const selection =
    ref.pick ??
    (ref.visibility === 'public'
      ? referenced.fields
          .filter(candidate => candidate.visibility !== 'private')
          .map(candidate => candidate.name)
      : undefined);
  if (selection) {
    const picked = selection.map(pickedName => {
      const match = referenced.fields.find(candidate => candidate.name === pickedName);
      if (!match)
        throw new Error(
          `${owner.name}.${fieldName} picks unknown field ${referenced.name}.${pickedName}`
        );
      return toSpec(match);
    });
    if (new Set(selection).size !== selection.length) {
      throw new Error(`${owner.name}.${fieldName} contains duplicate picked fields`);
    }
    nested = compileFields(
      {
        ...owner,
        name: `${owner.name}_${fieldName}`,
        schemaId: 0,
        storage: 'aos',
        variants: undefined,
      },
      picked,
      resolve,
      referenced.name
    );
  }
  return nested;
}

function validateVariants(owner: NetStructSpec, fields: readonly NetFieldLayout[]): void {
  const variants = owner.variants ?? [];
  if (variants.length === 0) return;
  const kind = fields.find(field => field.name === 'kind' && field.kind === 'scalar');
  if (!kind || (kind.type !== 'u8' && kind.type !== 'u16' && kind.type !== 'u32')) {
    throw new Error(`${owner.name} variants require a u8/u16/u32 kind field`);
  }
  const names = new Set<string>();
  const tags = new Set<number>();
  const fieldNames = new Set(fields.map(field => field.name));
  for (const variant of variants) {
    if (
      !variant.name ||
      names.has(variant.name) ||
      !Number.isSafeInteger(variant.tag) ||
      variant.tag < 0 ||
      tags.has(variant.tag)
    ) {
      throw new Error(`${owner.name} has an invalid or duplicate variant`);
    }
    for (const field of variant.fields) {
      if (!fieldNames.has(field)) {
        throw new Error(`${owner.name}.${variant.name} references unknown field ${field}`);
      }
    }
    names.add(variant.name);
    tags.add(variant.tag);
  }
}

function toSpec(field: NetFieldLayout): NetFieldSpec {
  const optional = field.optional ? { optional: field.optional } : {};
  if (field.kind === 'var') {
    return {
      id: field.id,
      name: field.name,
      type: field.type,
      visibility: field.visibility,
      ...optional,
    };
  }
  return field.kind === 'scalar'
    ? {
        id: field.id,
        name: field.name,
        type: field.enumValues ? { enum: field.enumValues } : field.type,
        count: field.count,
        visibility: field.visibility,
        ...optional,
      }
    : {
        id: field.id,
        name: field.name,
        type: field.type,
        count: field.count,
        visibility: field.visibility,
        ...optional,
      };
}

function fnv1a64(value: string): bigint {
  let hash = 0xcbf29ce484222325n;
  for (let i = 0; i < value.length; i++) {
    hash ^= BigInt(value.charCodeAt(i));
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash;
}
