/**
 * Measures how far a skinned body pulls apart at each joint.
 *
 *   npx tsx src/devtools/examples/_joint-gaps.mts <rig.glb>... -- <clip> <phase>
 *
 * "The shoulder looks disconnected" is not something to judge from a render:
 * the gap that matters is between the vertices one bone owns and the vertices
 * its neighbour owns, and at rest that distance is whatever the source
 * geometry already had. So report both — rest and posed — and the growth
 * between them is the tear.
 */
import { readFile } from 'node:fs/promises';
import { createPreviewSession } from '../session';

const argv = process.argv.slice(2);
const split = argv.indexOf('--');
const files = argv.slice(0, split < 0 ? undefined : split);
const clipName = split < 0 ? 'walk' : (argv[split + 1] ?? 'walk');
const phase = split < 0 ? 0.35 : Number(argv[split + 2] ?? 0.35);

const PAIRS: Array<[string, string]> = [
  ['clavicle_l', 'upperarm_l'], ['upperarm_l', 'lowerarm_l'], ['lowerarm_l', 'hand_l'],
  ['clavicle_r', 'upperarm_r'], ['upperarm_r', 'lowerarm_r'], ['lowerarm_r', 'hand_r'],
  ['pelvis', 'spine_01'], ['spine_01', 'spine_02'], ['spine_02', 'spine_03'],
  ['spine_03', 'neck_01'], ['neck_01', 'head'],
  ['spine_03', 'clavicle_l'], ['spine_03', 'clavicle_r'],
  ['thigh_l', 'calf_l'], ['thigh_r', 'calf_r'],
];

const session = await createPreviewSession({ width: 64, height: 64 });

for (const file of files) {
  await session.newScene({ clearColor: [0, 0, 0] });
  const container = await session.loadGlb(new Uint8Array(await readFile(file)), { id: 'rig' });
  const mesh = container.meshes.find((candidate: any) => candidate.skeleton) as any;
  const skeleton = mesh.skeleton;
  const groups = container.animationGroups ?? [];
  for (const group of groups) group.stop();

  const positions = mesh.getVerticesData('position') as Float32Array;
  const indices = mesh.getVerticesData('matricesIndices') as Float32Array;
  const weights = mesh.getVerticesData('matricesWeights') as Float32Array;
  const count = positions.length / 3;

  // Dominant bone per vertex: the tear is between the parts each bone OWNS,
  // and a vertex split 50/50 across the joint is exactly the one that hides it.
  const boneNames: string[] = skeleton.bones.map((bone: any) => bone.name);
  const owner = new Int32Array(count);
  for (let vertex = 0; vertex < count; vertex += 1) {
    let best = -1;
    let bestWeight = 0.5;
    for (let slot = 0; slot < 4; slot += 1) {
      const weight = weights[vertex * 4 + slot];
      if (weight > bestWeight) { bestWeight = weight; best = indices[vertex * 4 + slot]; }
    }
    owner[vertex] = best;
  }

  const skinned = (matrices: Float32Array) => {
    const out = new Float32Array(count * 3);
    for (let vertex = 0; vertex < count; vertex += 1) {
      let x = 0, y = 0, z = 0;
      for (let slot = 0; slot < 4; slot += 1) {
        const weight = weights[vertex * 4 + slot];
        if (weight <= 0) continue;
        const m = indices[vertex * 4 + slot] * 16;
        const px = positions[vertex * 3], py = positions[vertex * 3 + 1], pz = positions[vertex * 3 + 2];
        x += weight * (matrices[m] * px + matrices[m + 4] * py + matrices[m + 8] * pz + matrices[m + 12]);
        y += weight * (matrices[m + 1] * px + matrices[m + 5] * py + matrices[m + 9] * pz + matrices[m + 13]);
        z += weight * (matrices[m + 2] * px + matrices[m + 6] * py + matrices[m + 10] * pz + matrices[m + 14]);
      }
      out[vertex * 3] = x; out[vertex * 3 + 1] = y; out[vertex * 3 + 2] = z;
    }
    return out;
  };

  const gap = (points: Float32Array, a: number, b: number) => {
    const A: number[] = [], B: number[] = [];
    for (let vertex = 0; vertex < count; vertex += 1) {
      if (owner[vertex] === a) A.push(vertex);
      else if (owner[vertex] === b) B.push(vertex);
    }
    if (!A.length || !B.length) return NaN;
    let best = Infinity;
    // Sampled, not exhaustive: 20k x 20k is minutes, and the closest pair of a
    // stride-sampled set is within a vertex spacing of the true one.
    const strideA = Math.max(1, Math.floor(A.length / 400));
    const strideB = Math.max(1, Math.floor(B.length / 400));
    for (let i = 0; i < A.length; i += strideA) {
      const va = A[i] * 3;
      for (let k = 0; k < B.length; k += strideB) {
        const vb = B[k] * 3;
        const dx = points[va] - points[vb], dy = points[va + 1] - points[vb + 1], dz = points[va + 2] - points[vb + 2];
        const d = dx * dx + dy * dy + dz * dz;
        if (d < best) best = d;
      }
    }
    return Math.sqrt(best);
  };

  skeleton.returnToRest();
  skeleton.prepare(true);
  skeleton.computeAbsoluteMatrices(true);
  const restPoints = skinned(skeleton.getTransformMatrices(mesh));

  const clip = groups.find((group: any) => new RegExp(clipName, 'i').test(group.name)) ?? groups[0];
  const frame = clip.from + (clip.to - clip.from) * phase;
  clip.start(false, 1, frame, frame, false);
  clip.goToFrame(frame);
  clip.pause();
  skeleton.prepare(true);
  skeleton.computeAbsoluteMatrices(true);
  const posedPoints = skinned(skeleton.getTransformMatrices(mesh));

  const height = (() => {
    let lo = Infinity, hi = -Infinity;
    for (let vertex = 0; vertex < count; vertex += 1) {
      lo = Math.min(lo, restPoints[vertex * 3 + 1]);
      hi = Math.max(hi, restPoints[vertex * 3 + 1]);
    }
    return hi - lo;
  })();

  console.log(`\n${file}  (clip ${clip.name} @ ${frame.toFixed(1)}, body height ${height.toFixed(1)})`);
  console.log('  joint                        rest    posed    growth   (% of height)');
  for (const [a, b] of PAIRS) {
    const ia = boneNames.indexOf(a), ib = boneNames.indexOf(b);
    if (ia < 0 || ib < 0) continue;
    const rest = gap(restPoints, ia, ib);
    const posed = gap(posedPoints, ia, ib);
    if (!Number.isFinite(rest) || !Number.isFinite(posed)) continue;
    const growth = posed - rest;
    const flag = growth / height > 0.01 ? '  <-- tears' : '';
    console.log(
      `  ${(a + ' / ' + b).padEnd(28)}${rest.toFixed(2).padStart(6)}  ${posed.toFixed(2).padStart(6)}` +
      `  ${growth.toFixed(2).padStart(7)}   ${((growth / height) * 100).toFixed(2).padStart(6)}%${flag}`,
    );
  }
}

await session.dispose();
