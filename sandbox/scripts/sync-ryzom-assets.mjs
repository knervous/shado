#!/usr/bin/env node
/**
 * Stages merged Ryzom actors into the sandbox so the /ryzom pane can load them.
 *
 *   node scripts/sync-ryzom-assets.mjs            # curated subset
 *   node scripts/sync-ryzom-assets.mjs --all      # every built actor
 *   node scripts/sync-ryzom-assets.mjs --list
 *
 * Copies each actor GLB to public/shado/ryzom/ and writes ryzom-catalog.json,
 * which the playground turns into showcase models. GLBs are copied
 * uncompressed: the showcase's canonical loader takes a plain URL, and vite
 * serves these straight from public/.
 *
 * Ambient clips are chosen here rather than left to the showcase's own
 * heuristic, whose pattern is English (idle|walk|run|...) and matches nothing in
 * a Ryzom clip list — those are French (marche, course, attente).
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const sandboxRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.resolve(sandboxRoot, '..', '..');
const actorDir = path.join(repoRoot, 'assets/ryzom/converted/actor');
const outDir = path.join(sandboxRoot, 'public/shado/ryzom');

/** Locomotion and idle loops that read well as an ambient pose. */
const AMBIENT = /marche|course|idle|attente|debout|repos|stand|walk|run/i;
/** One-shots and states that look broken when held as an ambient loop. */
const UNSAFE = /mort|dead|death|die|ko_|coup|attack|atk|combat|frappe|saut|jump|nage|swim|assi|sit|touche|impact|degat/i;

/**
 * A curated default: two of each ethnic humanoid plus a spread of creatures.
 * Copying all 34 is ~0.5 GB into public/, which is a lot to hand a dev server
 * when most panes only need a handful.
 */
const CURATED = [
  'zorai_agents_actors_male',
  'zorai_agents_actors_female',
  'matis_agents_actors_male',
  'tryker_agents_actors_female',
  'caravan_agents_actors_male',
  'tryker_agents_monsters_cheval',
  'tryker_agents_monsters_kitin',
  'tryker_agents_monsters_chien',
  'tryker_agents_monsters_homins_degeneres_frahar',
];

/** Short, stable code from an actor id: ethnic + kind + gender. */
function codeFor(id) {
  const parts = id.split('_');
  const ethnic = parts[0].slice(0, 2);
  const tail = parts[parts.length - 1];
  const gender = tail === 'male' ? 'm' : tail === 'female' ? 'f' : '';
  const kind = id.includes('_monsters_') ? parts[parts.length - 1].slice(0, 4) : '';
  return (gender ? `${ethnic}-${gender}` : `${ethnic}-${kind || tail.slice(0, 6)}`).toLowerCase();
}

function labelFor(id) {
  return id
    .replace(/_agents_actors_/, ' ')
    .replace(/_agents_monsters_/, ' ')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

async function readActors() {
  const entries = await fs.readdir(actorDir).catch(() => []);
  const ids = entries.filter((f) => f.endsWith('.glb')).map((f) => f.slice(0, -4)).sort();
  const actors = [];
  for (const id of ids) {
    const sidecar = await fs
      .readFile(path.join(actorDir, `${id}.asset.json`), 'utf8')
      .then(JSON.parse)
      .catch(() => null);
    actors.push({ id, sidecar, bytes: (await fs.stat(path.join(actorDir, `${id}.glb`))).size });
  }
  return actors;
}

/** Clip names live in the GLB, so read them out of its JSON chunk. */
async function clipNames(glbPath) {
  const buf = await fs.readFile(glbPath);
  if (buf.readUInt32LE(0) !== 0x46546c67) return [];
  let off = 12;
  while (off < buf.length) {
    const len = buf.readUInt32LE(off);
    const type = buf.readUInt32LE(off + 4);
    off += 8;
    if (type === 0x4e4f534a) {
      const doc = JSON.parse(buf.subarray(off, off + len).toString('utf8'));
      return (doc.animations ?? []).map((a) => a.name).filter(Boolean);
    }
    off += len;
  }
  return [];
}

function pickAmbient(clips) {
  const safe = clips.filter((c) => AMBIENT.test(c) && !UNSAFE.test(c));
  if (safe.length) return safe.slice(0, 6);
  // Nothing matched: fall back to the first few, which still beats the
  // showcase picking a death animation as the resting pose.
  return clips.filter((c) => !UNSAFE.test(c)).slice(0, 4);
}

const all = process.argv.includes('--all');
const actors = await readActors();

if (!actors.length) {
  console.error(`no actors in ${path.relative(repoRoot, actorDir)} — run: npm run ryzom:actors`);
  process.exit(1);
}

if (process.argv.includes('--list')) {
  console.log(`${actors.length} actors:\n`);
  for (const a of actors) {
    const mark = CURATED.includes(a.id) ? '*' : ' ';
    console.log(`  ${mark} ${a.id.padEnd(52)} ${(a.bytes / 2 ** 20).toFixed(1).padStart(7)} MB  ${a.sidecar?.stats?.animations ?? 0} clips`);
  }
  console.log('\n* = in the curated default set');
  process.exit(0);
}

const selected = all ? actors : actors.filter((a) => CURATED.includes(a.id));
await fs.mkdir(outDir, { recursive: true });

const catalog = [];
let copied = 0;
for (const actor of selected) {
  const src = path.join(actorDir, `${actor.id}.glb`);
  const dest = path.join(outDir, `${actor.id}.glb`);
  await fs.copyFile(src, dest);
  copied += actor.bytes;

  const clips = await clipNames(src);
  const ambientClips = pickAmbient(clips);
  catalog.push({
    id: actor.id,
    code: codeFor(actor.id),
    label: labelFor(actor.id),
    kind: actor.id.includes('_monsters_') ? 'npc' : 'pc',
    clips: clips.length,
    ambientClips,
    meshes: actor.sidecar?.stats?.meshes ?? 0,
    triangles: actor.sidecar?.stats?.triangles ?? 0,
    joints: actor.sidecar?.stats?.joints ?? 0,
  });
  console.log(`  ${actor.id.padEnd(52)} ${clips.length} clips, ambient: ${ambientClips.slice(0, 3).join(', ') || '(none matched)'}`);
}

// Codes address models in the UI, so a collision would silently shadow an actor.
const seen = new Set();
for (const entry of catalog) {
  let code = entry.code;
  let n = 2;
  while (seen.has(code)) code = `${entry.code}${n++}`;
  seen.add(code);
  entry.code = code;
}

await fs.writeFile(path.join(outDir, 'ryzom-catalog.json'), JSON.stringify(catalog, null, 2) + '\n');
console.log(`\nstaged ${catalog.length} actors (${(copied / 2 ** 20).toFixed(1)} MB) -> ${path.relative(sandboxRoot, outDir)}`);
console.log('catalog: public/shado/ryzom/ryzom-catalog.json');
