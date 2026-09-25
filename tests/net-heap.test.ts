import { describe, expect, it } from '@jest/globals';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import ts from 'typescript';
import { compileNetLayouts, type NetStructSpec } from '../src/net/NetLayout';
import { emitNetStructModule } from '../src/net/emitNetStructModule';
import { emitNetStructRustModule } from '../src/net/emitNetStructRustModule';

const specs = [
  {
    name: 'Transform',
    layout: 'net',
    schemaId: 0x10,
    fields: [
      { id: 1, name: 'position', type: 'f32', count: 3 },
      { id: 2, name: 'flags', type: 'u8' },
    ],
  },
  {
    name: 'Modifier',
    layout: 'net',
    schemaId: 0x20,
    fields: [
      { id: 1, name: 'stat', type: 'str' },
      { id: 2, name: 'value', type: 'f32' },
      { id: 3, name: 'operation', type: 'u8' },
    ],
  },
  {
    name: 'Affix',
    layout: 'net',
    schemaId: 0x21,
    fields: [
      { id: 1, name: 'id', type: 'str' },
      { id: 2, name: 'kind', type: 'u8' },
      { id: 3, name: 'modifiers', type: { list: { struct: 'Modifier' } } },
    ],
  },
  {
    name: 'Item',
    layout: 'net',
    schemaId: 0x22,
    fields: [
      { id: 1, name: 'itemId', type: 'u32' },
      { id: 2, name: 'name', type: 'str' },
      { id: 3, name: 'icon', type: 'u16' },
      { id: 4, name: 'tags', type: { list: 'u16' } },
      { id: 5, name: 'weights', type: { list: 'f64' } },
      { id: 6, name: 'flags', type: { list: 'bool' } },
      { id: 7, name: 'blob', type: 'bytes' },
      { id: 8, name: 'transform', type: { struct: 'Transform' } },
      { id: 9, name: 'anchors', type: { list: { struct: 'Transform' } } },
      { id: 10, name: 'affixes', type: { list: { struct: 'Affix' } } },
      { id: 11, name: 'primary', type: { struct: 'Affix' } },
      { id: 12, name: 'pair', type: { struct: 'Affix', pick: ['id', 'kind'] }, count: 2 },
    ],
  },
] as const satisfies readonly NetStructSpec[];

const items = [
  {
    itemId: 1001,
    name: 'Ember Blade ⚔️ of Åsgard',
    icon: 42,
    tags: [3, 65535, 7],
    weights: [0.1, -2.5],
    flags: [true, false, true],
    blob: new Uint8Array([0, 255, 128]),
    transform: { position: [1.5, -2, 3], flags: 9 },
    anchors: [
      { position: [4, 5, 6], flags: 1 },
      { position: [-7, 8, -9], flags: 2 },
    ],
    affixes: [
      {
        id: 'burning',
        kind: 1,
        modifiers: [
          { stat: 'fire', value: 12.5, operation: 0 },
          { stat: 'haste', value: 0.25, operation: 1 },
        ],
      },
      { id: '', kind: 2, modifiers: [] },
    ],
    primary: { id: 'of the fox', kind: 2, modifiers: [{ stat: 'agi', value: 3, operation: 0 }] },
    pair: [
      { id: 'a', kind: 1 },
      { id: 'bb', kind: 2 },
    ],
  },
  {
    itemId: 2,
    name: '',
    icon: 0,
    tags: [],
    weights: [],
    flags: [],
    blob: new Uint8Array(),
    transform: { position: [0, 0, 0], flags: 0 },
    anchors: [],
    affixes: [],
    primary: { id: 'x', kind: 0, modifiers: [] },
    pair: [
      { id: '', kind: 0 },
      { id: 'z', kind: 3 },
    ],
  },
];

const rustc = spawnSync('rustc', ['--version']).status === 0;

async function loadTs(dir: string) {
  const js = ts.transpileModule(emitNetStructModule(specs), {
    compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  writeFileSync(path.join(dir, 'net.mjs'), js);
  return import(pathToFileURL(path.join(dir, 'net.mjs')).href);
}

describe('variable-length net fields', () => {
  it('keeps an 8-byte offset|count slot in the record and hashes the element shape', () => {
    const layouts = compileNetLayouts(specs);
    const item = layouts.get('Item')!;
    expect(item.variable).toBe(true);
    expect(layouts.get('Transform')!.variable).toBe(false);
    const name = item.fields.find(field => field.name === 'name')!;
    expect(name).toMatchObject({ kind: 'var', varKind: 'str', byteSize: 8, alignment: 4 });

    const changed = compileNetLayouts(
      specs.map(spec =>
        spec.name === 'Modifier'
          ? { ...spec, fields: [...spec.fields, { id: 4, name: 'extra', type: 'u8' as const }] }
          : spec
      )
    );
    expect(changed.get('Item')!.schemaHash).not.toBe(item.schemaHash);
  });

  it('rejects variable fields in SoA packets and as fixed arrays', () => {
    expect(() =>
      compileNetLayouts([
        {
          name: 'A',
          layout: 'net',
          schemaId: 1,
          storage: 'soa',
          fields: [{ id: 1, name: 's', type: 'str' }],
        },
      ])
    ).toThrow(/SoA/);
    expect(() =>
      compileNetLayouts([
        {
          name: 'A',
          layout: 'net',
          schemaId: 1,
          fields: [{ id: 1, name: 's', type: 'str', count: 2 } as never],
        },
      ])
    ).toThrow(/list/);
  });

  it('round-trips through the TypeScript views without copying scalar lists', async () => {
    const dir = mkdtempSync(path.join(tmpdir(), 'shado-net-heap-'));
    try {
      const net = await loadTs(dir);
      const bytes = net.encodeItemBatch(items) as Uint8Array;
      const batch = new net.ItemBatchView(bytes);
      expect(batch.count).toBe(2);
      const first = batch.record(0);
      expect(first.name).toBe(items[0]!.name);
      expect(first.weights).toBeInstanceOf(Float64Array);
      expect(first.weights.buffer).toBe(bytes.buffer);
      expect(Array.from(first.weights)).toEqual([0.1, -2.5]);
      expect(first.affixes(0).modifiers(1).stat).toBe('haste');
      expect(net.readItem(batch.record(0))).toEqual(items[0]);
      expect(net.readItem(batch.record(1))).toEqual(items[1]);
      expect(net.decodeItem(bytes)).toEqual(items[0]);

      const corrupt = bytes.slice();
      const nameSlot = compileNetLayouts(specs)
        .get('Item')!
        .fields.find(f => f.name === 'name')!;
      new DataView(corrupt.buffer).setUint32(32 + nameSlot.byteOffset, 1 << 20, true);
      expect(() => new net.ItemBatchView(corrupt).record(0).name).toThrow(RangeError);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  (rustc ? it : it.skip)(
    'writes and reads the same bytes in Rust, without allocating per field',
    async () => {
      const dir = mkdtempSync(path.join(tmpdir(), 'shado-net-heap-rs-'));
      try {
        const net = await loadTs(dir);
        const bytes = net.encodeItemBatch(items) as Uint8Array;
        const nameSlot = compileNetLayouts(specs)
          .get('Item')!
          .fields.find(f => f.name === 'name')!;
        writeFileSync(path.join(dir, 'items.bin'), bytes);
        writeFileSync(path.join(dir, 'net_structs.rs'), emitNetStructRustModule(specs));
        writeFileSync(
          path.join(dir, 'main.rs'),
          RUST_MAIN.replaceAll('NAME_SLOT', String(32 + nameSlot.byteOffset))
        );
        execFileSync('rustc', ['--edition', '2021', '-D', 'warnings', 'main.rs', '-o', 'heap'], {
          cwd: dir,
          stdio: 'pipe',
        });
        const output = execFileSync(path.join(dir, 'heap'), [dir], { encoding: 'utf8' });
        expect(output.trim()).toBe(Buffer.from(bytes).toString('hex'));
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    120_000
  );
});

const RUST_MAIN = `mod net_structs;
use net_structs::*;

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let burning = [
        Modifier { stat: "fire", value: 12.5, operation: 0 },
        Modifier { stat: "haste", value: 0.25, operation: 1 },
    ];
    let fox = [Modifier { stat: "agi", value: 3.0, operation: 0 }];
    let affixes = [
        Affix { id: "burning", kind: 1, modifiers: &burning },
        Affix { id: "", kind: 2, modifiers: &[] },
    ];
    let anchors = [
        Transform { position: [4.0, 5.0, 6.0], flags: 1 },
        Transform { position: [-7.0, 8.0, -9.0], flags: 2 },
    ];
    let items = [
        Item {
            item_id: 1001,
            name: "Ember Blade ⚔️ of Åsgard",
            icon: 42,
            tags: &[3, 65535, 7],
            weights: &[0.1, -2.5],
            flags: &[true, false, true],
            blob: &[0, 255, 128],
            transform: Transform { position: [1.5, -2.0, 3.0], flags: 9 },
            anchors: &anchors,
            affixes: &affixes,
            primary: Affix { id: "of the fox", kind: 2, modifiers: &fox },
            pair: [ItemPair { id: "a", kind: 1 }, ItemPair { id: "bb", kind: 2 }],
        },
        Item {
            item_id: 2,
            primary: Affix { id: "x", ..Default::default() },
            pair: [ItemPair::default(), ItemPair { id: "z", kind: 3 }],
            ..Default::default()
        },
    ];

    // One buffer, reused: the second encode must not need to grow it.
    let mut out = Vec::new();
    encode_net_messages_into(&items, &mut out);
    let capacity = out.capacity();
    encode_net_messages_into(&items, &mut out);
    assert_eq!(out.capacity(), capacity);

    let bytes = std::fs::read(format!("{}/items.bin", dir)).unwrap();
    assert_eq!(out, bytes);

    let batch = ItemView::decode_batch(&bytes).unwrap();
    assert_eq!(batch.len(), 2);
    let first = batch.get(0).unwrap();
    assert_eq!(first.item_id(), 1001);
    assert_eq!(first.name(), "Ember Blade ⚔️ of Åsgard");
    assert_eq!(first.tags().iter().collect::<Vec<_>>(), vec![3, 65535, 7]);
    assert_eq!(first.weights().get(1), Some(-2.5));
    assert_eq!(first.flags().iter().collect::<Vec<_>>(), vec![true, false, true]);
    assert_eq!(first.blob(), &[0, 255, 128]);
    assert_eq!(first.transform(), items[0].transform);
    assert_eq!(first.anchors().get(1), Some(anchors[1]));
    let affix = first.affixes().get(0).unwrap();
    assert_eq!(affix.id(), "burning");
    assert_eq!(affix.modifiers().get(1).unwrap().stat(), "haste");
    assert_eq!(affix.modifiers().get(1).unwrap().value(), 0.25);
    assert!(first.affixes().get(1).unwrap().modifiers().is_empty());
    assert_eq!(first.primary().modifiers().get(0).unwrap().stat(), "agi");
    assert_eq!(first.pair()[1].id(), "bb");
    assert_eq!(batch.get(1).unwrap().pair()[1].id(), "z");
    assert_eq!(batch.get(1).unwrap().name(), "");

    // A slot pointing past the heap, and text that is not UTF-8, fail up front.
    let mut corrupt = bytes.clone();
    corrupt[NAME_SLOT..NAME_SLOT + 4].copy_from_slice(&(1u32 << 20).to_le_bytes());
    assert_eq!(ItemView::decode_batch(&corrupt).err(), Some(NetDecodeError::HeapOutOfBounds));
    let mut corrupt = bytes.clone();
    let name_at = u32::from_le_bytes(corrupt[NAME_SLOT..NAME_SLOT + 4].try_into().unwrap()) as usize;
    let heap = net_heap_offset(&bytes);
    corrupt[heap + name_at] = 0xff;
    assert_eq!(ItemView::decode_batch(&corrupt).err(), Some(NetDecodeError::InvalidUtf8));
    assert_eq!(ItemView::decode_batch(&bytes[..40]).err(), Some(NetDecodeError::InvalidLength));
    assert_eq!(AffixView::decode(&bytes).err(), Some(NetDecodeError::SchemaMismatch));

    print!("{}", out.iter().map(|b| format!("{:02x}", b)).collect::<String>());
}

fn net_heap_offset(bytes: &[u8]) -> usize {
    let stride = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
    let count = u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize;
    32 + (stride * count + 7) / 8 * 8
}
`;
