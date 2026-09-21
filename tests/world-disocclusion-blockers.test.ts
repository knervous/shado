import { describe, expect, it } from '@jest/globals';

import {
  captureFrame,
  DisocclusionAdmission,
  decodeDisocclusionSidecar,
  wordsToBytes,
  clusterRegionDigest,
  compileFixtureWorld,
  DISOCCLUSION_FIXTURES,
  sha256Hex,
  stampTargetBounds,
  verifyPrototypeManifest,
  verifySidecarInputs,
  worldIdentity,
  zoneBakeGeometry,
  type DisocclusionSidecar,
} from '../src/world/disocclusion';
import { SHADO_OCCLUDER_ELIGIBILITY_REVISION } from '../src/world/occluder-scene';
import type { ShadoWorldSpatialPackage } from '../src/world/types';

/**
 * pvs.md V3: a disocclusion row is only as good as the blockers it was baked
 * against, and in a town most blockers are placed-object prototypes. A
 * prototype-only edit (same spatial package, same world GLB) must force
 * reference admission; so must anything the reader cannot verify.
 */

const encode = (text: string) => new TextEncoder().encode(text);
const CHURCH = '/eqrequiem/objects/church/final.glb.gz';
const WALL = '/eqrequiem/objects/wall/final.glb.gz';

async function setup() {
  const world = compileFixtureWorld(DISOCCLUSION_FIXTURES['two-room']!()) as ShadoWorldSpatialPackage;
  // The fixture has no stamps; give it two referenced prototypes.
  (world as any).objects = { ...(world as any).objects, prototypes: { source: [WALL, CHURCH, WALL] } };
  const files = new Map<string, Uint8Array>([
    [CHURCH, encode('church v1')],
    [WALL, encode('wall v1')],
  ]);
  const manifest = {
    eligibilityRevision: SHADO_OCCLUDER_ELIGIBILITY_REVISION,
    files: [
      { source: CHURCH, sha256: await sha256Hex(files.get(CHURCH)!) },
      { source: WALL, sha256: await sha256Hex(files.get(WALL)!) },
    ],
  };
  const spatial = encode('spatial');
  const glb = encode('glb');
  const sidecar = {
    meta: {
      world: worldIdentity(world),
      inputs: {
        clusterRegionSha256: await clusterRegionDigest(world),
        zone: {
          name: 'fixture',
          spatialSha256: await sha256Hex(spatial),
          glbSha256: await sha256Hex(glb),
          prototypesSha256: 'x',
          prototypeFiles: 2,
          prototypes: manifest,
        },
      },
    },
    words: new Uint32Array(0),
  } as unknown as DisocclusionSidecar;
  const resolve = async (source: string) => files.get(source) ?? null;
  const options = { eligibilityRevision: SHADO_OCCLUDER_ELIGIBILITY_REVISION, release: 'r1' };
  return { world, files, manifest, sidecar, spatial, glb, resolve, options };
}

describe('disocclusion blocker validation (V3)', () => {
  it('accepts matching content and refuses a prototype-only change with the same spatial and GLB bytes', async () => {
    const { world, files, sidecar, spatial, glb, resolve, options } = await setup();
    expect(await verifySidecarInputs(sidecar, world, { spatial, glb, prototypes: { resolve, ...options } })).toBeNull();
    files.set(CHURCH, encode('church v2 -- a new doorway'));
    expect(await verifySidecarInputs(sidecar, world, { spatial, glb, prototypes: { resolve, ...options } })).toMatch(
      /church.*content differs/
    );
  });

  it('treats unverified blocker content as unsupported', async () => {
    const { world, files, manifest, sidecar, spatial, glb, resolve, options } = await setup();
    // No resolver at all.
    expect(await verifySidecarInputs(sidecar, world, { spatial, glb })).toMatch(/not verified/);
    // A prototype the client cannot find.
    files.delete(WALL);
    expect(await verifyPrototypeManifest(world, manifest, resolve, options)).toMatch(/wall.*does not resolve/);
    // A resolver that fails outright.
    expect(await verifyPrototypeManifest(world, manifest, async () => Promise.reject(new Error('503')), options)).toMatch(
      /could not be resolved/
    );
    // A sidecar baked before the manifest existed.
    const old = structuredClone(sidecar.meta) as any;
    delete old.inputs.zone.prototypes;
    expect(
      await verifySidecarInputs({ ...sidecar, meta: old }, world, { spatial, glb, prototypes: { resolve, ...options } })
    ).toMatch(/no prototype manifest/);
  });

  it('refuses different eligibility rules, an incomplete manifest and a file that appeared after the bake', async () => {
    const { world, manifest, resolve, options } = await setup();
    expect(await verifyPrototypeManifest(world, manifest, resolve, { ...options, eligibilityRevision: 'other' })).toMatch(
      /eligibility/
    );
    expect(await verifyPrototypeManifest(world, { ...manifest, files: manifest.files.slice(1) }, resolve, options)).toMatch(
      /does not cover/
    );
    const missingAtBake = { ...manifest, files: manifest.files.map((f) => (f.source === WALL ? { ...f, sha256: null } : f)) };
    expect(await verifyPrototypeManifest(world, missingAtBake, resolve, options)).toMatch(/missing at bake time/);
  });

  it('a prototype missing at bake survives serialization and verifies only while it stays missing (A4)', async () => {
    const fixture = DISOCCLUSION_FIXTURES['two-room']!();
    const world = compileFixtureWorld(fixture) as ShadoWorldSpatialPackage;
    (world as any).objects = { ...(world as any).objects, prototypes: { source: [CHURCH, WALL] } };
    const files = new Map<string, Uint8Array>([[WALL, encode('wall v1')]]);
    const spatial = encode('spatial');
    const glb = encode('glb');
    const identity = worldIdentity(world);
    const wordsPerRow = Math.ceil((identity.width * identity.height) / 32);
    const payload = wordsToBytes(new Uint32Array(wordsPerRow));
    const meta = {
      format: 'eltania-disocclusion-pvs',
      version: 1,
      experimental: true,
      generator: { name: 'shado-disocclusion', revision: 'test' },
      createdAt: '2026-09-21T00:00:00.000Z',
      world: identity,
      inputs: {
        clusterRegionSha256: await clusterRegionDigest(world),
        zone: {
          name: 'fixture',
          spatialSha256: await sha256Hex(spatial),
          glbSha256: await sha256Hex(glb),
          prototypesSha256: 'x',
          prototypeFiles: 2,
          prototypes: {
            eligibilityRevision: SHADO_OCCLUDER_ELIGIBILITY_REVISION,
            files: [
              { source: CHURCH, sha256: null },
              { source: WALL, sha256: await sha256Hex(files.get(WALL)!) },
            ],
          },
        },
      },
      settings: {},
      domains: [{ id: 'd0', capture: fixture.captures[0]!, extTan: [1, 1], viewcellHalf: [1, 1], wordOffset: 0, counts: {}, timings: {} }],
      payload: { endianness: 'little', wordsPerRow, wordCount: wordsPerRow, sha256: await sha256Hex(payload) },
    };
    // Through the real serialized boundary, not a hand-built object.
    const sidecar = await decodeDisocclusionSidecar(JSON.stringify(meta), payload);
    const verify = (resolve: (source: string) => Promise<Uint8Array | null>) =>
      verifySidecarInputs(sidecar, world, {
        spatial,
        glb,
        prototypes: { resolve, eligibilityRevision: SHADO_OCCLUDER_ELIGIBILITY_REVISION, release: 'r1' },
      });
    // Missing at bake and at runtime: the bake treated it as unknown, so do we.
    expect(await verify(async (source) => files.get(source) ?? null)).toBeNull();
    // Missing at bake, present now: drift, refused.
    expect(await verify(async (source) => (source === CHURCH ? encode('church') : files.get(source) ?? null))).toMatch(
      /missing at bake time but resolves now/
    );
    // Null anywhere else is still corruption.
    const corrupt = { ...meta, world: { ...meta.world, width: null } };
    await expect(decodeDisocclusionSidecar(JSON.stringify(corrupt), payload)).rejects.toThrow(/meta\.world\.width is null/);
  });

  it('refuses a changed LOD selection or chain: target bounds were the union of those exact levels (A1)', async () => {
    const { world, files, manifest, resolve, options } = await setup();
    const SELECTION = '/eqrequiem/objects/object-lods.json';
    const CHAIN = '/eqrequiem/objects/wall/final.lods.glb.gz';
    const lodFiles = new Map([...files, [SELECTION, encode('{"levels":{"wall":[1]}}')], [CHAIN, encode('wall lods v1')]]);
    const withLods = {
      ...manifest,
      lods: {
        manifest: { source: SELECTION, sha256: await sha256Hex(lodFiles.get(SELECTION)!) },
        files: [{ source: CHAIN, sha256: await sha256Hex(lodFiles.get(CHAIN)!) }],
      },
    };
    const from = (map: Map<string, Uint8Array>) => async (source: string) => map.get(source) ?? null;
    expect(await verifyPrototypeManifest(world, withLods, from(lodFiles), options)).toBeNull();
    const newChain = new Map(lodFiles).set(CHAIN, encode('wall lods v2'));
    expect(await verifyPrototypeManifest(world, withLods, from(newChain), options)).toMatch(/LOD chain .* differs/);
    const newSelection = new Map(lodFiles).set(SELECTION, encode('{"levels":{}}'));
    expect(await verifyPrototypeManifest(world, withLods, from(newSelection), options)).toMatch(/LOD selection .* differs/);
    void resolve;
  });

  it('hashes each prototype once per release, and again for a new release', async () => {
    const { world, manifest, files, options } = await setup();
    let fetches = 0;
    const counting = async (source: string) => {
      fetches++;
      return files.get(source) ?? null;
    };
    const cache = new Map<string, string | null>();
    expect(await verifyPrototypeManifest(world, manifest, counting, { ...options, cache })).toBeNull();
    expect(await verifyPrototypeManifest(world, manifest, counting, { ...options, cache })).toBeNull();
    expect(fetches).toBe(2);
    expect(await verifyPrototypeManifest(world, manifest, counting, { ...options, release: 'r2', cache })).toBeNull();
    expect(fetches).toBe(4);
  });

  it('names a cluster whose package bounds disagree with its geometry, and never lets it block', () => {
    const fixture = DISOCCLUSION_FIXTURES['two-room']!();
    const world = compileFixtureWorld(fixture) as ShadoWorldSpatialPackage;
    const parts = world.primitives.map((primitive) => {
      const source = fixture.primitives.find((p) => p.name === primitive.name)!;
      const hash = primitive.name.lastIndexOf('#');
      return {
        node: hash >= 0 ? primitive.name.slice(0, hash) : primitive.name,
        positions: source.positions,
        indices: source.indices,
        doubleSided: true,
        exclusion: null,
      };
    });
    const clean = zoneBakeGeometry(world, parts, []);
    expect(clean.manifest.frameMismatches).toBe(0);
    expect(clean.manifest.suspectClusters).toEqual([]);

    // The Crownward case: the package's sphere for a cluster sits far from
    // the geometry the runtime actually draws.
    world.clusters.centerY[0] = world.clusters.centerY[0]! - 130;
    const skewed = zoneBakeGeometry(world, parts, []);
    expect(skewed.manifest.suspectClusters.map((s) => s.cluster)).toEqual([0]);
    expect(skewed.manifest.suspectClusters[0]!.worstDistanceRadii).toBeGreaterThan(1);
    const { triangleTarget, blocker } = skewed.geometry;
    let clusterTriangles = 0;
    for (let t = 0; t < triangleTarget.length; t++) {
      if (triangleTarget[t] !== 0) continue;
      clusterTriangles++;
      expect(blocker![t]).toBe(0);
    }
    expect(clusterTriangles).toBeGreaterThan(0);
    expect(skewed.manifest.blockers.total).toBe(clean.manifest.blockers.total - clusterTriangles);
  });

  it('stamps are targets: opaque triangles carry the stamp id, a non-blocking box carries its whole bound', () => {
    const fixture = DISOCCLUSION_FIXTURES['two-room']!();
    const world = compileFixtureWorld(fixture) as ShadoWorldSpatialPackage;
    const parts = world.primitives.map((primitive) => {
      const source = fixture.primitives.find((p) => p.name === primitive.name)!;
      const hash = primitive.name.lastIndexOf('#');
      return { node: hash >= 0 ? primitive.name.slice(0, hash) : primitive.name, positions: source.positions, indices: source.indices, doubleSided: true, exclusion: null };
    });
    const clusters = world.clusters.firstIndex.length;
    // Stamp 0 has an opaque triangle and a larger drawn bound; stamp 1 must never be hidden.
    const wall = { name: 'stamp-0:wall', stamp: 0, material: 'm', positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), indices: new Uint32Array([0, 1, 2]) };
    const bounds = new Float32Array([-1, -1, -1, 2, 3, 4, NaN, NaN, NaN, NaN, NaN, NaN]);
    const { geometry, manifest } = zoneBakeGeometry(world, parts, [wall], bounds);
    expect(manifest.stampTargets).toEqual({ stamps: 2, targets: 1, neverHidden: 1 });
    expect(geometry.stampTargetBase).toBe(clusters);
    expect(geometry.stampTargetCount).toBe(2);
    const own: number[] = [];
    for (let t = 0; t < geometry.triangleTarget.length; t++) if (geometry.triangleTarget[t] === clusters) own.push(t);
    // 1 blocker triangle + 12 box triangles, only the first one blocks.
    expect(own.length).toBe(13);
    expect(own.filter((t) => geometry.blocker![t]).length).toBe(1);
    expect(geometry.triangleTarget.includes(clusters + 1)).toBe(false);
  });

  it('target bounds are the union of every drawn level; an unreadable chain is never hidden (A1)', () => {
    // Three stamps at x = 0, 10, 20, identity rotation, scale 2 on the third.
    const n = 3;
    const col = (values: number[]) => values;
    const world = {
      objects: {
        prototypes: { id: ['wall', 'wall-lod-fails', 'wall'], source: ['/w.glb.gz', '/f.glb.gz', '/w.glb.gz'] },
        stamps: {
          id: ['a', 'b', 'c'],
          prototype: col([0, 1, 2]),
          enabled: col([1, 1, 1]),
          phaseMask: col([0, 0, 0]),
          positionX: col([0, 10, 20]),
          positionY: col([0, 0, 0]),
          positionZ: col([0, 0, 0]),
          rotationX: col([0, 0, 0]),
          rotationY: col([0, 0, 0]),
          rotationZ: col([0, 0, 0]),
          scaleX: col([1, 1, 2]),
          scaleY: col([1, 1, 2]),
          scaleZ: col([1, 1, 2]),
        },
      },
    } as unknown as ShadoWorldSpatialPackage;
    // Level 0 is a unit cube; its coarse level bulges 1.5 units higher -- a
    // simplifier is free to move a silhouette outward.
    const level0 = new Float32Array([0, 0, 0, 1, 1, 1]);
    const coarse = new Float32Array([0, 0, 0, 1, 2.5, 1]);
    const bounds = stampTargetBounds(world, (prototype) => (prototype === 1 ? null : [level0, coarse]))!;
    const box = (s: number) => Array.from(bounds.slice(s * 6, s * 6 + 6));
    // The union, to float tolerance -- not level 0 plus a guessed margin.
    expect(box(0)[4]).toBeGreaterThanOrEqual(2.5);
    expect(box(0)[4]).toBeLessThan(2.5 + 0.01);
    expect(box(0)[1]).toBeLessThanOrEqual(0);
    expect(box(0)[1]).toBeGreaterThan(-0.01);
    // Scale applies to the union.
    expect(box(2)[4]).toBeGreaterThanOrEqual(5);
    expect(box(2)[4]).toBeLessThan(5.01);
    // An unreadable chain: never a target.
    expect(box(1).every(Number.isNaN)).toBe(true);
    // And the geometry gives the enlarged target its whole box.
    const fixture = DISOCCLUSION_FIXTURES['two-room']!();
    const fixtureWorld = compileFixtureWorld(fixture) as ShadoWorldSpatialPackage;
    const parts = fixtureWorld.primitives.map((primitive) => {
      const source = fixture.primitives.find((p) => p.name === primitive.name)!;
      const hash = primitive.name.lastIndexOf('#');
      return { node: hash >= 0 ? primitive.name.slice(0, hash) : primitive.name, positions: source.positions, indices: source.indices, doubleSided: true, exclusion: null };
    });
    const { geometry } = zoneBakeGeometry(fixtureWorld, parts, [], bounds);
    const base = geometry.stampTargetBase!;
    let top = -Infinity;
    for (let t = 0; t < geometry.triangleTarget.length; t++) {
      if (geometry.triangleTarget[t] !== base) continue;
      for (let v = 0; v < 3; v++) top = Math.max(top, geometry.positions[geometry.indices[t * 3 + v]! * 3 + 1]!);
    }
    expect(top).toBeGreaterThanOrEqual(2.5);
  });

  it('a version-3 sidecar carries per-stamp and relied-on rows; admission unions them and refuses a stamp-count mismatch', async () => {
    const fixture = DISOCCLUSION_FIXTURES['two-room']!();
    const world = compileFixtureWorld(fixture) as ShadoWorldSpatialPackage;
    const stampCount = 37; // not a multiple of 32: padding bits matter
    (world as any).objects = { prototypes: { source: [] }, stamps: { id: Array.from({ length: stampCount }, (_, i) => `s${i}`) } };
    const capture = fixture.captures[0]!;
    const identity = worldIdentity(world);
    const regions = identity.width * identity.height;
    const wordsPerRow = Math.ceil(regions / 32);
    const stampWords = Math.ceil(stampCount / 32);
    const clusters = world.clusters.firstIndex.length;
    const clusterWords = Math.ceil(clusters / 32);
    const reliedClusterOffset = wordsPerRow + stampWords;
    const reliedStampOffset = reliedClusterOffset + clusterWords;
    const words = new Uint32Array(reliedStampOffset + stampWords);
    words.fill(0xffffffff, 0, wordsPerRow);
    if (regions % 32) words[wordsPerRow - 1] = ((1 << (regions % 32)) - 1) >>> 0;
    // Admit stamps 0, 5 and 36 only; the rows rely on stamp 5 and cluster 0.
    for (const st of [0, 5, 36]) words[wordsPerRow + (st >>> 5)]! |= (1 << (st & 31)) >>> 0;
    words[reliedStampOffset]! |= 1 << 5;
    words[reliedClusterOffset]! |= 1;
    const payload = wordsToBytes(words);
    const meta = {
      format: 'eltania-disocclusion-pvs',
      version: 3,
      experimental: true,
      generator: { name: 'shado-disocclusion', revision: 'test' },
      createdAt: '2026-09-21T00:00:00.000Z',
      world: identity,
      inputs: { clusterRegionSha256: await clusterRegionDigest(world) },
      stamps: { count: stampCount, wordsPerRow: stampWords },
      relied: { clusters, clusterWordsPerRow: clusterWords },
      settings: {},
      domains: [
        {
          id: 'd0',
          capture,
          extTan: [1, 1],
          viewcellHalf: [1, 1],
          wordOffset: 0,
          stampWordOffset: wordsPerRow,
          reliedClusterWordOffset: reliedClusterOffset,
          reliedStampWordOffset: reliedStampOffset,
          counts: {},
          timings: {},
        },
      ],
      payload: { endianness: 'little', wordsPerRow, wordCount: words.length, sha256: await sha256Hex(payload) },
    };
    const sidecar = await decodeDisocclusionSidecar(JSON.stringify(meta), payload);
    const frame = captureFrame(capture);
    const c = [0, 1, 2].map((a) => (capture.sourceMin[a]! + capture.sourceMax[a]!) / 2) as [number, number, number];
    const ray = (dx: number, dy: number) => [0, 1, 2].map((a) => frame.forward[a]! + frame.right[a]! * dx + frame.up[a]! * dy) as [number, number, number];
    const pose = { position: c, cornerRays: [ray(-0.01, -0.01), ray(0.01, -0.01), ray(-0.01, 0.01), ray(0.01, 0.01)] };
    const result = new DisocclusionAdmission(world, sidecar).evaluate(pose);
    expect(result.mode).toBe('baked');
    expect(result.admittedStamps).toBe(3);
    expect([...result.stampMask!].flatMap((v, i) => (v ? [i] : []))).toEqual([0, 5, 36]);
    expect([...result.reliedStamps!].flatMap((v, i) => (v ? [i] : []))).toEqual([5]);
    expect([...result.reliedClusters!].flatMap((v, i) => (v ? [i] : []))).toEqual([0]);

    // Rows without the relied-on blockers cannot be applied safely (A1).
    const v2Words = words.slice(0, wordsPerRow + stampWords);
    const v2Bytes = wordsToBytes(v2Words);
    const { relied: _relied, ...v2Meta } = meta;
    const v2 = await decodeDisocclusionSidecar(
      JSON.stringify({
        ...v2Meta,
        version: 2,
        domains: [{ ...meta.domains[0], reliedClusterWordOffset: undefined, reliedStampWordOffset: undefined }],
        payload: { ...meta.payload, wordCount: v2Words.length, sha256: await sha256Hex(v2Bytes) },
      }),
      v2Bytes
    );
    expect(new DisocclusionAdmission(world, v2).evaluate(pose)).toMatchObject({ mode: 'reference', reason: expect.stringMatching(/no relied-on blocker rows/) });

    // A stamp bit past the count is corrupt.
    const bad = words.slice();
    bad[wordsPerRow + 1]! |= 1 << 10;
    const badBytes = wordsToBytes(bad);
    await expect(
      decodeDisocclusionSidecar(JSON.stringify({ ...meta, payload: { ...meta.payload, sha256: await sha256Hex(badBytes) } }), badBytes)
    ).rejects.toThrow(/stamp padding/);
    // A world with a different stamp count cannot use these rows.
    (world as any).objects.stamps.id.push('extra');
    expect(new DisocclusionAdmission(world, sidecar).evaluate(pose)).toMatchObject({ mode: 'reference', stampMask: null });
  });
});
