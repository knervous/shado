/**
 * Render the artifacts the shado runtime actually consumes.
 *
 *   npx tsx src/devtools/examples/shado-runtime-assets.mts <zone> [outDir]
 *
 * A zone ships as a runtime GLB plus a compiled spatial package, and the bake
 * writes a separate vertex-lighting field. This loads the runtime GLB the way
 * the client would, reads the spatial package for its framing and diagnostics,
 * and optionally paints a lighting field over the geometry — so a bake can be
 * proved against the exact artifact that ships.
 *
 * It is ordinary Babylon plus shado's own world module; the session only
 * supplies the engine.
 */
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { join } from 'node:path';

import { createPreviewSession } from '../session';

const [zone = 'talioscrownward', outDir = './shado-previews/runtime'] = process.argv.slice(2);
const root = new URL('../../../../', import.meta.url).pathname;
const worlds = `${root}client/public/eqrequiem/worlds`;

const inflate = async (path: string) => {
  const raw = await readFile(path);
  return path.endsWith('.gz') ? gunzipSync(raw) : raw;
};

const glb = new Uint8Array(await inflate(`${worlds}/${zone}.glb.gz`));
const spatial = JSON.parse((await inflate(`${worlds}/${zone}.spatial.json.gz`)).toString('utf8'));
console.log(
  `  spatial package: ${spatial.triangleCount} triangles, ${spatial.clusters.radius.length} clusters, `
  + `${spatial.renderChunks.primitive.length} render chunks, source ${spatial.source}`,
);

const field = process.env.FIELD ? JSON.parse(await readFile(process.env.FIELD, 'utf8')) : null;

const session = await createPreviewSession({ width: 640, height: 480 });
try {
  await session.newScene({ clearColor: [0.02, 0.02, 0.03] });
  await session.loadGlb(glb, { id: zone });

  if (field?.meshes) {
    // The field is keyed by glTF MESH name, while Babylon names its meshes
    // after the glTF NODE. Matching them directly hits only the handful where
    // the two names coincide, so resolve node -> mesh through the GLB's own
    // JSON. A multi-primitive glTF mesh becomes `<node>_primitive<N>`, which is
    // the same convention preprocess/world-core relies on.
    const view = new DataView(glb.buffer, glb.byteOffset, glb.byteLength);
    const gltf = JSON.parse(new TextDecoder().decode(glb.subarray(20, 20 + view.getUint32(12, true))).trimEnd());
    const nameByBabylonMesh = new Map<string, string>();
    (gltf.nodes ?? []).forEach((node: any, nodeIndex: number) => {
      if (!Number.isInteger(node.mesh)) return;
      const gltfMesh = gltf.meshes[node.mesh];
      const nodeName = node.name || `node${nodeIndex}`;
      const primitives = gltfMesh.primitives ?? [];
      primitives.forEach((_primitive: unknown, primitiveIndex: number) => {
        const babylonName = primitives.length === 1 ? nodeName : `${nodeName}_primitive${primitiveIndex}`;
        nameByBabylonMesh.set(babylonName, gltfMesh.name);
      });
    });

    const { VertexBuffer } = await import('@babylonjs/core/Buffers/buffer.js');
    let painted = 0;
    let missing = 0;
    for (const mesh of session.scene.meshes) {
      const fieldKey = nameByBabylonMesh.get(mesh.name) ?? mesh.name;
      const colors = field.meshes[fieldKey];
      if (!colors) { if (mesh.getTotalVertices?.()) missing++; continue; }
      mesh.setVerticesData(VertexBuffer.ColorKind, Array.from(colors) as never, false, 4);
      mesh.useVertexColors = true;
      painted++;
    }
    console.log(`  painted the baked field onto ${painted} meshes (${missing} had no entry)`);
  }

  for (const view of ['iso', 'top'] as const) {
    await session.frameCamera({ view, zoom: 2.0 });
    const file = join(outDir, `${zone}.${view}.png`);
    await session.captureToFile(file);
    console.log(`  ${file}`);
  }
} finally {
  await session.dispose();
}
