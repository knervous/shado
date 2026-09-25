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
    name: 'Tag',
    layout: 'net',
    fields: [
      { id: 1, name: 'name', type: 'str' },
      { id: 2, name: 'weight', type: 'u8', optional: true },
    ],
  },
  {
    name: 'Combat',
    layout: 'net',
    schemaId: 0x30,
    fields: [
      { id: 1, name: 'targetId', type: 'u32' },
      { id: 2, name: 'outcome', type: { enum: ['hit', 'miss', 'crit'] } },
      { id: 3, name: 'verb', type: 'str' },
      { id: 4, name: 'nextSwingInMs', type: 'u32', optional: true },
      { id: 5, name: 'effect', type: 'str', optional: true },
      { id: 6, name: 'weapon', type: 'str', optional: 'null' },
      { id: 7, name: 'tags', type: { list: 'str' } },
      { id: 8, name: 'school', type: { enum: ['fire', 'frost'] }, optional: true },
      { id: 9, name: 'tag', type: { struct: 'Tag' }, optional: true },
      { id: 10, name: 'notes', type: { list: { struct: 'Tag' } } },
    ],
  },
  {
    name: 'Vitals',
    layout: 'net',
    schemaId: 0x31,
    fields: [
      { id: 1, name: 'hp', type: 'i32' },
      { id: 2, name: 'maxHp', type: 'i32', optional: true },
      { id: 3, name: 'stance', type: { enum: ['stand', 'sit', 'self'] } },
    ],
  },
  {
    name: 'Pose',
    layout: 'net',
    schemaId: 0x32,
    storage: 'soa',
    fields: [
      { id: 1, name: 'kind', type: 'u8' },
      { id: 2, name: 'mood', type: { enum: ['calm', 'angry'] } },
      { id: 3, name: 'hp', type: 'i32' },
    ],
  },
] as const satisfies readonly NetStructSpec[];

const full = {
  targetId: 77,
  outcome: 'crit',
  verb: 'smites',
  nextSwingInMs: 1200,
  effect: 'burn ✦',
  weapon: 'Ember Blade',
  tags: ['a', '', 'ünïcode'],
  school: 'frost',
  tag: { name: 'boss', weight: 9 },
  notes: [{ name: 'x' }, { name: 'y', weight: 3 }],
};

const partial = {
  targetId: 5,
  outcome: 'miss',
  verb: '',
  weapon: null,
  tags: [],
  notes: [],
};

const rustc = spawnSync('rustc', ['--version']).status === 0;

async function loadTs(dir: string) {
  const js = ts.transpileModule(emitNetStructModule(specs), {
    compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  writeFileSync(path.join(dir, 'net.mjs'), js);
  return import(pathToFileURL(path.join(dir, 'net.mjs')).href);
}

describe('optional, enum and string-list net fields', () => {
  it('reserves presence words, stores enums as indices, and keeps nested-only records out of the registry', () => {
    const layouts = compileNetLayouts(specs);
    const combat = layouts.get('Combat')!;
    expect(combat.presenceWords).toBe(1);
    expect(combat.fields[0]).toMatchObject({ name: 'targetId', byteOffset: 4 });
    expect(combat.fields.find(field => field.name === 'outcome')).toMatchObject({
      type: 'u8',
      enumValues: ['hit', 'miss', 'crit'],
    });
    expect(combat.fields.filter(field => field.presenceBit !== undefined).map(f => f.name)).toEqual(
      ['nextSwingInMs', 'effect', 'weapon', 'school', 'tag']
    );
    expect(layouts.get('Pose')!.presenceWords).toBe(0);
    expect(layouts.get('Tag')!.schemaId).toBe(0);

    const source = emitNetStructModule(specs);
    expect(source).not.toContain('TAG_SCHEMA_ID');
    expect(source).toContain('[COMBAT_SCHEMA_ID]: {');
    expect(source).not.toContain('[POSE_SCHEMA_ID]');

    const rust = emitNetStructRustModule(specs);
    expect(rust).toContain('pub enum CombatOutcome {');
    expect(rust).toContain('pub next_swing_in_ms: Option<u32>,');
    expect(rust).toContain("pub tags: &'a [&'a str],");
    expect(rust).not.toContain('TAG_SCHEMA_ID');
  });

  it('rejects optional fields in SoA packets and malformed enums', () => {
    const soa = [
      {
        name: 'Bad',
        layout: 'net',
        schemaId: 1,
        storage: 'soa',
        fields: [{ id: 1, name: 'hp', type: 'i32', optional: true }],
      },
    ] as const satisfies readonly NetStructSpec[];
    expect(() => compileNetLayouts(soa)).toThrow(/optional/);
    const dup = [
      {
        name: 'Bad',
        layout: 'net',
        schemaId: 1,
        fields: [{ id: 1, name: 'mode', type: { enum: ['a', 'a'] } }],
      },
    ] as const satisfies readonly NetStructSpec[];
    expect(() => compileNetLayouts(dup)).toThrow(/unique/);
  });

  it('round-trips absent, null and present values through TypeScript', async () => {
    const dir = mkdtempSync(path.join(tmpdir(), 'shado-net-optional-'));
    try {
      const net = await loadTs(dir);
      expect(net.decodeCombat(net.encodeCombat(full))).toEqual(full);
      const decoded = net.decodeCombat(net.encodeCombat(partial));
      expect(decoded).toEqual(partial);
      expect('nextSwingInMs' in decoded).toBe(false);
      expect(decoded.weapon).toBeNull();

      const vitals = net.encodeVitals({ hp: 10, stance: 'sit' });
      expect(net.decodeVitals(vitals)).toEqual({ hp: 10, stance: 'sit' });
      const view = net.viewVitals(vitals);
      expect(view.hasMaxHp).toBe(false);
      view.maxHp = 50;
      expect(view.hasMaxHp).toBe(true);
      expect(() => {
        view.stance = 'fly';
      }).toThrow(RangeError);

      const bytes = net.encodeCombat(full);
      expect(net.peekNetSchemaId(bytes)).toBe(0x30);
      expect(net.NET_PACKET_CODECS[0x31].decode(vitals)).toEqual({
        hp: 10,
        maxHp: 50,
        stance: 'sit',
      });
      expect(net.NET_PACKET_CODECS[0x30].decode(net.NET_PACKET_CODECS[0x30].encode(full))).toEqual(
        full
      );
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  (rustc ? it : it.skip)(
    'writes the same bytes from Rust and reads TypeScript packets back',
    async () => {
      const dir = mkdtempSync(path.join(tmpdir(), 'shado-net-optional-rust-'));
      try {
        const net = await loadTs(dir);
        const pose = net.createPoseBatch(2);
        pose.kind.set([1, 2]);
        pose.mood.set([1, 0]);
        pose.hp.set([-4, 90]);
        const packets = {
          full: net.encodeCombat(full) as Uint8Array,
          partial: net.encodeCombat(partial) as Uint8Array,
          vitals: net.encodeVitals({ hp: -3, maxHp: 40, stance: 'sit' }) as Uint8Array,
          pose: pose.bytes as Uint8Array,
        };
        for (const [name, bytes] of Object.entries(packets)) {
          writeFileSync(path.join(dir, `${name}.bin`), bytes);
        }
        writeFileSync(path.join(dir, 'net_structs.rs'), emitNetStructRustModule(specs));
        writeFileSync(path.join(dir, 'main.rs'), RUST_MAIN);
        execFileSync('rustc', ['--edition', '2021', '-D', 'warnings', 'main.rs', '-o', 'opt'], {
          cwd: dir,
          stdio: 'pipe',
        });
        const output = execFileSync(path.join(dir, 'opt'), [dir], { encoding: 'utf8' });
        const hex = (bytes: Uint8Array) => Buffer.from(bytes).toString('hex');
        expect(output.trim().split('\n')).toEqual([
          hex(packets.full),
          hex(packets.partial),
          hex(packets.vitals),
          hex(packets.pose),
        ]);
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

fn read(dir: &str, name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/{}.bin", dir, name)).unwrap()
}

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let notes = [Tag { name: "x", weight: None }, Tag { name: "y", weight: Some(3) }];
    let full = Combat {
        target_id: 77,
        outcome: CombatOutcome::Crit,
        verb: "smites",
        next_swing_in_ms: Some(1200),
        effect: Some("burn ✦"),
        weapon: Some("Ember Blade"),
        tags: &["a", "", "ünïcode"],
        school: Some(CombatSchool::Frost),
        tag: Some(Tag { name: "boss", weight: Some(9) }),
        notes: &notes,
    };
    let partial = Combat { target_id: 5, outcome: CombatOutcome::Miss, ..Combat::default() };

    let bytes = read(&dir, "full");
    let view = CombatView::decode(&bytes).unwrap();
    assert_eq!(view.outcome(), CombatOutcome::Crit);
    assert_eq!(view.outcome().as_str(), "crit");
    assert_eq!(view.next_swing_in_ms(), Some(1200));
    assert_eq!(view.effect(), Some("burn ✦"));
    assert_eq!(view.school(), Some(CombatSchool::Frost));
    assert_eq!(view.tags().iter().collect::<Vec<_>>(), vec!["a", "", "ünïcode"]);
    assert_eq!(view.tag().unwrap().weight(), Some(9));
    assert_eq!(view.notes().get(0).unwrap().weight(), None);
    assert_eq!(view.notes().get(1).unwrap().name(), "y");
    println!("{}", hex(&full.encode()));

    let bytes = read(&dir, "partial");
    let view = CombatView::decode(&bytes).unwrap();
    assert_eq!(view.next_swing_in_ms(), None);
    assert_eq!(view.weapon(), None);
    assert!(view.tag().is_none());
    assert!(view.tags().is_empty());
    println!("{}", hex(&partial.encode()));

    let vitals = Vitals { hp: -3, max_hp: Some(40), stance: VitalsStance::Sit };
    assert_eq!(Vitals::decode(&read(&dir, "vitals")).unwrap(), vitals);
    assert_eq!(VitalsStance::Self_.as_str(), "self");
    println!("{}", hex(&vitals.encode()));

    let pose = PoseBatch::decode(&read(&dir, "pose")).unwrap();
    assert_eq!(pose.row(0).mood, PoseMood::Angry);
    assert_eq!(PoseMood::from_raw(9), PoseMood::Calm);
    println!("{}", hex(&PoseBatch::from_rows(&[pose.row(0), pose.row(1)]).encode()));
}
`;
