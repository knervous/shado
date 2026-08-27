# Migrating a variant supermesh to module draws

For models that ship every equipment variant as its own submesh and hide the
unworn ones per instance — the Ryzom-style `equip -> submesh` layout. This is
the pattern that stops scaling somewhere around a hundred actors.

Written for a coding agent to follow end to end. Every claim here is from a
shipped migration; the traps in §6 were each expensive to find once.

---

## 1. Why it stops scaling

A variant supermesh draws one merged mesh per model and hides the unselected
variants per instance, usually by writing a sentinel into a per-instance
appearance array that the vertex shader turns into a degenerate position.

The hiding happens **after** the vertex has been skinned. Every instance
therefore pays full skinning for every variant in the wardrobe and throws most
of it away. Cost is `instances x total wardrobe vertices`, no matter how little
of it is visible.

Measured on a real humanoid wardrobe (43 submeshes, 17 variants across 7 body
pieces): each drawn actor skinned 9,406 vertices to display 3,762. That is 60%
of the vertex work discarded, for every actor, every frame. A deeper wardrobe
is worse — 24k vertices across 96 submeshes wastes about 85%.

Module draws invert the ownership: each `(piece, variant)` group becomes its own
mesh that draws **only the actors wearing it**. Cost becomes
`sum over modules of (module vertices x actors wearing that module)`, which
scales with what is actually visible.

p95 frame ms, WebGPU, M-series, same scene both ways:

| actors | merged supermesh | module draws |
|---:|---:|---:|
| 1,000 | 18.25 | 18.36 |
| 10,000 | 66.97 | 18.12 |
| 20,000 | 134.98 | 30.03 |

The two are equivalent at low counts — the split is not a micro-optimization,
it changes the slope. Note the shape: merged scales with
`population x wardrobe size`, modules with visible geometry.

## 2. What you keep

Nothing about the actor arena changes. One container, one visibility pass, one
set of actor records, one skeleton, one VAT, one appearance array. Only draw
ownership moves. In particular:

- **Keep your per-instance variant sentinel.** It is the bucketing source, it
  keeps any un-migrated path working, and the in-shader clip becomes a harmless
  no-op for geometry that is no longer in the draw.
- **Keep your submesh ordinals global** across all modules. If your appearance
  array is indexed `submeshIndex + actor * submeshCount`, it still is. Nothing
  outside the split needs to know the geometry moved.

## 3. The API

```ts
import { splitMeshesIntoModules, ShadoModuleDrawSet } from '@knervous/shado';
```

**Split the geometry.** Group by whatever key selects a variant, and name the
custom vertex streams that must survive the merge:

```ts
const geometry = splitMeshesIntoModules(sourceMeshes, {
  groupKey: (mesh, index) => `${piece(index)}:${variant(index)}`,
  preserveAttributes: [{ kind: 'submeshData', stride: 2 }],
  name: (key) => `${model}#${key}`,
});
```

Returning a constant key collapses everything into one module, which is
byte-for-byte the supermesh you started with. That is the useful baseline: ship
it first, confirm nothing moved, then turn on the real key.

**Own the draws.**

```ts
const draws = new ShadoModuleDrawSet(engine, geometry);
draws.registerThinInstanceAttribute('matrix', 16);
```

**Refresh once per frame**, wherever your visibility already lands:

```ts
draws.refresh(container.visibleActorIndices, (actorIndex, moduleIndex) =>
  showsModule(actorIndex, moduleIndex),
);
```

`refresh` walks the visible list once per module, skips the GPU upload for any
bucket that came out element-wise unchanged, and switches an empty module off
entirely — which also keeps it out of GPU picking. It returns
`ShadoModuleDrawStats` with `submittedVertices`, `baselineVertices` and
`vertexWorkReduction`, which is what you report to prove the migration.

**Bind per draw.** In each module material's per-draw bind, after binding the
arena, replace its global visible list with the module's own:

```ts
const drawn = draws.bindSelection(moduleIndex, effect);
mesh.forcedInstanceCount = drawn;
```

**Instances fan out**, so the thin-instance count stays aligned with the arena:

```ts
const index = draws.addThinInstance(matrix);   // same index on every module
draws.setThinInstanceMatrixAt(index, matrix);
```

## 4. Where membership comes from

An actor shows a module when the module's first submesh is not hidden. Derive
that from the array you already write — do not invent a second source of truth,
or buckets and pixels will disagree in ways that only show up on one machine.

The cheapest correct shape is a byte per `(actor, submesh)` mirrored at the one
place your appearance writes funnel through:

```ts
setAppearance(actor, submesh, count, slice, ...) {
  arena.setAppearance(actor * count + submesh, [slice, ...]);
  this.submeshVisible[actor * count + submesh] = slice >= 0 ? 1 : 0;   // mirror
}
```

Zero that actor's row when a slot is recycled, or a new instance inherits the
outfit the previous occupant was wearing.

## 5. Order of work

1. Split with a constant `groupKey`. One module, everything unchanged. Ship it.
2. Turn on the real key. Materials still shared — expect the WebGPU breakage in
   §6 if you skip step 3.
3. One material per module.
4. Wire `refresh` into your visibility pass and after any equipment change.
5. Fan out thin instances and picking registration.
6. Verify with §7 before touching anything else.

## 6. Traps

**One material per module, not one shared.** A module's compact index list is a
per-draw binding. On WebGPU that binding lives in the material's draw context,
which is shared by every mesh using that material — so one shared material
across module meshes is last-bind-wins and modules draw each other's actors.
Clone the material per module. They share one compiled effect when the defines
match, so the compile cost is paid once; 17-40 materials per model is nothing.

**`MergeMeshes` drops your custom vertex streams.** It rebuilds vertex data
through `VertexData.ExtractFromMesh`, which only knows Babylon's own kinds.
That is what `preserveAttributes` is for. If you merge by hand, concatenate them
yourself in merge order.

**`MergeMeshes` skips the winding flip for a one-mesh group.** For its root mesh
it only reverses a reflected source's winding inside the branch that resizes the
index buffer to fit the meshes being appended — so a group of exactly one mesh
keeps the wrong winding. This is invisible while a whole model merges as one
supermesh, and appears the moment a split produces a single-submesh module
(hair, typically). It reads as inside-out **and unlit**, because backface
culling then draws the interior, whose normals face away from the light.
`splitMeshesIntoModules` repairs this; `repairSingleSourceMergeWinding` is
exported if you merge by hand.

**A stale texture makes a material silently never draw.** Babylon's
`Material.isReady()` returns false if *any* texture bound to the material is not
ready, and a **disposed** texture is not ready. The material then never becomes
ready, its effect is never created, and it never draws — with no shader error
and no console warning. The symptom reads as "unlit" or "black", which sends you
into the lighting code. If geometry vanishes with a clean console, probe
`Object.entries(material._textures).map(([k, t]) => [k, t.isReady()])` first.
Anything that seeds a material with a texture it may later replace must replay
that binding.

**Module meshes are typed structurally.** `ShadoModuleMesh` is the slice of
Babylon's `Mesh` this API touches, not `Mesh` itself, so your own
`@babylonjs/core` install satisfies it. Narrow back with a cast at the boundary
when you need Babylon-specific members:

```ts
const mesh = module.mesh as unknown as Mesh;   // your Babylon, your Mesh
```

**Check your appearance array is not being truncated (Shado < 1.4.0).**
`getVarArrayCount` read its stride as `floatStride | 1` — a bitwise OR, not a
default — so every *even* stride reported 4/5 of the real element count for a
`vec4`. `writeVarArrayRange` compares against that count, decides the array is
short, and calls `resizeVarArray`, which shrinks and zero-fills the tail. The
result is that everything you write past ~80% occupancy silently disappears.
On an appearance array indexed `actor * submeshCount + submesh`, that means the
last fifth of the crowd renders with a zeroed appearance — black, untextured,
or wearing slot 0 — and nothing is logged. It hides during a migration because
small test crowds never reach 80%. Fixed in 1.4.0; on an older version, probe
it before blaming the split:

```ts
const needed = actorCount * submeshCount;
const have = container.getVarArrayCount('appearance');   // must be >= needed
```

**A deep texture array cannot generate mips on WebGPU.** If each variant's
albedo is a layer of a `RawTexture2DArray`, mip generation fails past a
handful of layers and every layer that failed samples black. The tell is that
the *nearest* actors look right and the rest of the crowd is black — mip level
0 is the layer you uploaded, so only minified samples are wrong. Build deep
arrays with `useMipChain = false`; the cost is aliasing at distance, not
correctness.

**Do not frustum-cull the module meshes.** Actor transforms live in the arena,
so the Babylon source mesh has no authoritative world bounds. Set
`alwaysSelectAsActiveMesh = true` and let the visibility pass own culling.

**Refresh after equipment changes, not only after culling.** Anything that
rewrites the appearance array changes bucket membership. A per-frame refresh
covers it; a detached renderer that runs no cull observer (a paperdoll or
inventory preview) must call `refresh` itself.

## 7. Verifying

The invariant: **for each piece, the module draw counts must sum to the pool's
visible actor count.** Every actor wears exactly one variant per piece, so if
they do not sum, membership is wrong.

```ts
const stats = draws.refresh(container.visibleActorIndices, isMember);
// stats.vertexWorkReduction ~ 1 / (visible fraction of the wardrobe)
```

Expect a submitted-vertex reduction of roughly `1 - visible/total`. The
reference migration measured 3,762 of 9,406 (2.50x) for a curated wardrobe and
4,259 of 10,451 (2.45x) for its second body.

Judge by these numbers, not by pixels: a module that draws the right actors and
a module that draws all of them look identical on one actor.

Black or untextured actors are almost never the split. The split changes *which
actors* a draw submits, never what a vertex looks like, so a wrong bucket shows
as a missing or duplicated garment, not as a black one. Walk the appearance
array directly before reading any module code — count how many entries came out
zeroed, and where they start:

```ts
// Read the segment, not getVarArrayPtr - that reports 0 for a var-array.
const seg = (container as any)._varSeg.appearance;
const values = container.arena.view(seg.offF, seg.lenF);
let zeroed = 0;
for (let i = 0; i < actorCount * submeshCount; i++) {
  if (values[i * 4 + 1] === 0) zeroed++;   // channel 1 of your appearance vec4
}
```

A count that is a clean fraction of the total, starting at a fixed index, is a
storage bug (see the var-array trap in §6). A count scattered across actors is a
membership bug.

## 8. Worked example

`src/showcase/ShadoSupermeshModuleDemo.ts` is a runnable end-to-end slice: it
builds a variant supermesh, migrates it, and reports the work reduction. Read it
alongside §3.

For the same pattern on a real asset rather than a synthetic one, the sandbox
route `/hum-wardrobe` loads a 43-submesh Ryzom-derived humanoid, splits it into
17 modules, and scales to 10,000 actors with an on-screen overlay of actors,
draw calls and submitted vertices. `sandbox/src/HumWardrobePlayground.ts` is
about 550 lines and contains every step of §5 in order.
