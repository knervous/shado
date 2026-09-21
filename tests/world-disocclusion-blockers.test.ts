import { describe, expect, it } from '@jest/globals';

import {
  clusterRegionDigest,
  compileFixtureWorld,
  DISOCCLUSION_FIXTURES,
  sha256Hex,
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
});
