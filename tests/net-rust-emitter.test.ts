import { describe, expect, it } from '@jest/globals';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import ts from 'typescript';
import type { NetStructSpec } from '../src/net/NetLayout';
import { emitNetStructModule } from '../src/net/emitNetStructModule';
import { emitNetStructRustModule } from '../src/net/emitNetStructRustModule';

const specs = [
  {
    name: 'Transform',
    layout: 'net',
    schemaId: 0x10,
    fields: [
      { id: 1, name: 'position', type: 'f32', count: 3 },
      { id: 2, name: 'rotation', type: 'f32', count: 4 },
      { id: 3, name: 'flags', type: 'u8' },
      { id: 4, name: 'hidden', type: 'bool', visibility: 'private' },
    ],
  },
  {
    name: 'Sample',
    layout: 'net',
    schemaId: 0x11,
    fields: [
      { id: 1, name: 'id', type: 'u32' },
      { id: 2, name: 'tiny', type: 'i8' },
      { id: 3, name: 'big', type: 'u64' },
      { id: 4, name: 'signedBig', type: 'i64' },
      { id: 5, name: 'delta', type: 'i16' },
      { id: 6, name: 'ratio', type: 'f64' },
      { id: 7, name: 'alive', type: 'bool' },
      { id: 8, name: 'type', type: 'u16' },
      { id: 9, name: 'transform', type: { struct: 'Transform' } },
      { id: 10, name: 'anchors', type: { struct: 'Transform', pick: ['position'] }, count: 2 },
      { id: 11, name: 'bytes', type: 'u8', count: 3 },
    ],
  },
  {
    name: 'Snapshot',
    layout: 'net',
    schemaId: 0x12,
    storage: 'soa',
    variants: [
      { name: 'player', tag: 1, fields: ['state'] },
      { name: 'npc', tag: 2, fields: ['state', 'hp'] },
    ],
    fields: [
      { id: 1, name: 'kind', type: 'u8' },
      { id: 2, name: 'entityId', type: 'u32' },
      { id: 3, name: 'state', type: { struct: 'Transform', visibility: 'public' } },
      { id: 4, name: 'hp', type: 'i32' },
    ],
  },
] as const satisfies readonly NetStructSpec[];

const rustc = spawnSync('rustc', ['--version']).status === 0;

describe('rust net struct emitter', () => {
  it('names records, projections, packets and variants after the TypeScript module', () => {
    const source = emitNetStructRustModule(specs);

    expect(source).toContain('pub struct Sample {');
    expect(source).toContain('pub signed_big: i64,');
    expect(source).toContain('pub r#type: u16,');
    expect(source).toContain('pub anchors: [SampleAnchors; 2],');
    expect(source).toContain('pub struct SampleAnchors {');
    expect(source).toContain('impl NetPacket for Sample {');
    expect(source).toContain('pub struct SnapshotBatch {');
    expect(source).toContain('pub state_position: Vec<f32>,');
    expect(source).not.toContain('pub state_hidden');
    expect(source).toContain('pub const KIND_NPC: u8 = 2;');
    expect(source).not.toContain('impl NetPacket for Snapshot {');
    expect(source).toContain('use std::vec::Vec;');
    expect(emitNetStructRustModule(specs, { vec: 'alloc' })).toContain('use alloc::vec::Vec;');
  });

  it('shares schema hashes with the TypeScript module', () => {
    const tsSource = emitNetStructModule(specs);
    const rustSource = emitNetStructRustModule(specs);
    for (const name of ['TRANSFORM', 'SAMPLE', 'SNAPSHOT']) {
      const hash = new RegExp(`${name}_SCHEMA_HASH(?:: u64)? = (0x[0-9a-f]+)`);
      expect(rustSource.match(hash)?.[1]).toBe(tsSource.match(hash)?.[1]);
    }
  });

  (rustc ? it : it.skip)(
    'encodes and decodes the same bytes as the TypeScript module',
    async () => {
      const dir = mkdtempSync(path.join(tmpdir(), 'shado-net-rust-'));
      try {
        const js = ts.transpileModule(emitNetStructModule(specs), {
          compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 },
        }).outputText;
        writeFileSync(path.join(dir, 'net.mjs'), js);
        const net = await import(pathToFileURL(path.join(dir, 'net.mjs')).href);

        const transform = {
          position: [1.5, -2.25, 3],
          rotation: [0, 0.5, -0.5, 1],
          flags: 7,
          hidden: true,
        };
        const sample = net.encodeSample({
          id: 0xdeadbeef,
          tiny: -5,
          big: 0xfedcba9876543210n,
          signedBig: -1234567890123n,
          delta: -300,
          ratio: 0.1,
          alive: true,
          type: 65535,
          transform,
          anchors: [{ position: [4, 5, 6] }, { position: [-7, -8, -9] }],
          bytes: [1, 2, 255],
        }) as Uint8Array;

        const snapshot = net.createSnapshotBatch(2);
        snapshot.kind.set([1, 2]);
        snapshot.entityId.set([10, 20]);
        snapshot.statePosition.set([1, 2, 3, 4, 5, 6]);
        snapshot.stateRotation.set([0, 0, 0, 1, 0.5, 0.5, 0.5, 0.5]);
        snapshot.stateFlags.set([3, 4]);
        snapshot.hp.set([-1, 99]);
        const snapshotBytes = snapshot.bytes as Uint8Array;

        writeFileSync(path.join(dir, 'sample.bin'), sample);
        writeFileSync(path.join(dir, 'snapshot.bin'), snapshotBytes);
        writeFileSync(path.join(dir, 'net_structs.rs'), emitNetStructRustModule(specs));
        writeFileSync(path.join(dir, 'main.rs'), RUST_MAIN);
        execFileSync(
          'rustc',
          ['--edition', '2021', '-D', 'warnings', 'main.rs', '-o', 'parity'],
          { cwd: dir, stdio: 'pipe' }
        );
        const output = execFileSync(path.join(dir, 'parity'), [dir], { encoding: 'utf8' });
        const [sampleBuilt, sampleRoundTrip, snapshotBuilt, snapshotRoundTrip] = output
          .trim()
          .split('\n');

        const hex = (bytes: Uint8Array) => Buffer.from(bytes).toString('hex');
        expect(sampleBuilt).toBe(hex(sample));
        expect(sampleRoundTrip).toBe(hex(sample));
        expect(snapshotBuilt).toBe(hex(snapshotBytes));
        expect(snapshotRoundTrip).toBe(hex(snapshotBytes));
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    120_000
  );
});

const RUST_MAIN = `mod net_structs;
use net_structs::*;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let transform = Transform {
        position: [1.5, -2.25, 3.0],
        rotation: [0.0, 0.5, -0.5, 1.0],
        flags: 7,
        hidden: true,
    };
    let sample = Sample {
        id: 0xdeadbeef,
        tiny: -5,
        big: 0xfedcba9876543210,
        signed_big: -1234567890123,
        delta: -300,
        ratio: 0.1,
        alive: true,
        r#type: 65535,
        transform,
        anchors: [
            SampleAnchors { position: [4.0, 5.0, 6.0] },
            SampleAnchors { position: [-7.0, -8.0, -9.0] },
        ],
        bytes: [1, 2, 255],
    };
    let sample_bytes = std::fs::read(format!("{}/sample.bin", dir)).unwrap();
    let decoded = Sample::decode(&sample_bytes).unwrap();
    assert_eq!(decoded, sample);
    assert_eq!(Sample::decode(&sample_bytes[..40]), Err(NetDecodeError::InvalidLength));
    assert_eq!(peek_net_schema_id(&sample_bytes), Some(SAMPLE_SCHEMA_ID));
    println!("{}", hex(&sample.encode()));
    println!("{}", hex(&decoded.encode()));

    let rows = [
        Snapshot {
            kind: Snapshot::KIND_PLAYER,
            entity_id: 10,
            state: SnapshotState { position: [1.0, 2.0, 3.0], rotation: [0.0, 0.0, 0.0, 1.0], flags: 3 },
            hp: -1,
        },
        Snapshot {
            kind: Snapshot::KIND_NPC,
            entity_id: 20,
            state: SnapshotState { position: [4.0, 5.0, 6.0], rotation: [0.5, 0.5, 0.5, 0.5], flags: 4 },
            hp: 99,
        },
    ];
    let snapshot_bytes = std::fs::read(format!("{}/snapshot.bin", dir)).unwrap();
    let decoded = SnapshotBatch::decode(&snapshot_bytes).unwrap();
    assert_eq!(decoded.row(1), rows[1]);
    assert_eq!(SnapshotBatch::decode(&sample_bytes), Err(NetDecodeError::SchemaMismatch));
    println!("{}", hex(&SnapshotBatch::from_rows(&rows).encode()));
    println!("{}", hex(&decoded.encode()));
}
`;
