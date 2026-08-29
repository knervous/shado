# Shado Sandbox

React + Vite Babylon.js sandbox for exercising the local `@knervous/shado`
package.

## Development

```bash
npm install
npm run dev
```

The sandbox depends on the parent package as `@knervous/shado: "file:.."`. Vite and
TypeScript use the package `source` export condition, so edits in `../src`
are consumed directly without running `npm run dev` or `npm run build` in the
parent package first. Symlink preservation ensures both projects resolve the
same Babylon.js installation.

## What It Shows

- Babylon.js 9 scene setup
- `ShadoInstanceContainer` actor instancing
- DQ/VAT mesh rendering
- WASM-backed frustum culling
- MSDF nameplate rendering backed by `NameplateData` and `@knervous/shado/msdf`
- Lean dynamic-entity rendering, movement, picking, and batched mutations
- Drag-and-drop ingestion and worker/WASM VAT baking for animated GLB files
- Per-variant module draws over a shared arena on a real humanoid wardrobe

Open `http://localhost:5173/` after starting the dev server.

## Humanoid wardrobe module draws

The demonstration for `splitMeshesIntoModules` / `ShadoModuleDrawSet` on a real
asset. Open:

```text
http://localhost:5173/hum-wardrobe?renderer=babylonjs&backend=webgpu&model=hum
```

`model=hum|huf` swaps the whole bundle; both bodies are vendored side by side.
It is also linked from the nav bar as **Wardrobe modules**, and is part of the
deployed build — its assets are tracked, and the SPA rewrite is a catch-all, so
the route works on the published site as well as locally.

The asset is a Ryzom-derived body promoted into the Eltania client: 74
submeshes grouped into 35 `(piece, variation)` modules across 7 body pieces
(`hn ch ft lg ua hr he`) — four civilian wardrobes, three armour tiers and
seven hairstyles. Every actor lives in one packed arena with one appearance
var-array and one VAT texture; only draw ownership is split.

### What the page can vary, and what it costs

Everything below is free at the draw level. The proof panel is the point: turn
all of it on and the draw count does not move, because none of it is geometry.

| Lever | Mechanism | Draws added |
| --- | --- | --- |
| Outfit (7 pieces x 3-7 variations) | picks the module | 0 — modules already exist |
| Role uniform (12) | repaints the sheet a mesh already samples | 0 |
| Face (8 minted) | one atlas layer on the visage primitive | 0 |
| Complexion (8) | multiply where the response marks skin | 0 |
| Heraldic device | the response's green channel | 0 |
| Dye | multiply where the response marks dyeable | 0 |

A **role uniform** owns no mesh. Variations `00-03` are the tier ladder, `04-19`
are uniform repaints and `20+` are grafted meshes the ladder can never reach, so
selecting a uniform moves only the atlas slice a submesh samples. Two of the
twelve ship minted art (`04` Talios Watch, `05` Lowharbor Dockwatch); the other
ten are **palette only** — a faction that differs by colour needs no generated
art at all, which is the claim the dropdown lets you check.

A minted uniform also **pins the tier its layers were painted from**. The
repaint reuses the sheet a mesh already samples, so tier-2 chain paint on a
tier-0 tunic is not a recolour, it is the wrong UVs.

The **material response** is what makes a dye mean something. It is a second
atlas, one mask per albedo layer: `r` marks skin, `g` the heraldic charge, `b`
what a dye may recolour. Without it a dye is a flat multiply that repaints the
face and hands too. It is uploaded lossless while the albedo is JPEG, because it
is read as three thresholds rather than looked at — a softened edge bleeds dye
onto a face and puts a charge off-centre.

**What the overlay proves.** Actors, draw calls, populated modules, fps, frame
ms and the submitted-vs-baseline vertex ratio update together. Step the crowd
from 1 to 10,000 and the draw count does not move: it is one per populated
module, not one per instance. At the default 65% variability all 17
modules are populated, and 10,000 actors draw 17 times for 37.1M submitted
vertices against 94.1M for the same wardrobe merged as one supermesh — a 2.54x
saving.
Drop variability to 0% and the crowd collapses onto the fewest modules, which
is the honest lower bound — fewer draws, but everyone dressed alike.

**Controls.** Crowd size and outfit variability; a per-piece variation select
and dye swatch; `Randomize tints` for a distinct per-actor per-piece dye (drawn
in HSL, because uniform random RGB averages to mud and swamps the albedo);
`Reset` to return to white; and a clip select. Tint costs nothing at the draw
level — it rides in the appearance `vec4` that is already being written, so the
draw count is identical dyed or plain.

Each actor animates from its own clip and its own phase, so the crowd does not
move as one organism. That is per-instance state in the arena, not per-instance
draws.

**Assets** live in `public/shado/<model>/` and are vendored by
`client/scripts/vendor-hum-wardrobe-demo.mjs` in the private Eltania repo:

```bash
node client/scripts/vendor-hum-wardrobe-demo.mjs --model hum --side 256
```

It reads the **post-map** `.babylon`, because only that artifact carries the
`{piece, variation, texNum}` stamps the wardrobe map produces, and throws
rather than emitting an unstamped bundle. Each bundle is the gzipped
`.babylon`, the gzipped matrix VAT the page skins from, a `.svat` (Shado
dual-quaternion VAT — vendored for parity with the client, not yet what this
page consumes), a JPEG grid sheet atlas, a lossless PNG grid sheet of the
material response, and a `.wardrobe.json` manifest naming the pieces,
variations, clips, runtime scale/yaw and the role vocabulary.

The manifest's `appearance` block — uniforms, palettes, devices, complexions,
per-trade compositions — is read out of the game's own
`Game/Constants/role-bodies.ts` at vendor time rather than restated in the demo,
so the page cannot drift from the table the game ships.

The albedo is JPEG at quality 92 with no chroma subsampling. The wardrobe grew
from 43 layers to 104 and the lossless sheet went with it, 3.9 MB to 11; at q92
it is 2.1 MB for a mean error of 1.58/255, with 0.45% of texels off by more than
8. The response sheet stays PNG for the reason given above, and is half the
albedo's resolution — it is a low-frequency mask, and matching the albedo would
have doubled the page weight for detail nothing samples.

Two things about this page are load-bearing traps rather than choices: the
atlas is uploaded **without a mip chain**, because WebGPU's
`RawTexture2DArray` mip generation fails for deep arrays and every layer that
fails samples black at distance; and the `.babylon` is handed to Babylon
through a blob URL rather than a `data:` URL, which parses in a fraction of the
time at this size.

## NM_M supermesh scalability benchmark

Sync the generated humanoid assets, start the sandbox, and open the benchmark
route with the full Babylon.js renderer:

```bash
npm run sync:supermesh
npm run dev
```

Open the actor editor at:

```text
http://localhost:5173/supermesh-scale?renderer=babylonjs&model=nm-m-supermesh&mode=explore
```

With no `path` the route runs the newest translation the active backend
supports — `cached` on WebGPU, `hybrid` on WebGL2 — so the default exercises the
module buckets, and on WebGPU the compute pre-skin cache and resolved pose
palette. The title bar names the active path and what it exercises. A saved
`path=cached` URL opened on WebGL2 degrades to `hybrid` rather than failing.

Every bucket receives only the actors selecting that part, while all buckets
share one packed actor arena and one VAT texture. Each actor animates from its
own clip and phase; only `cached` holds a single cohort pose, because it deforms
the module library once per pose and rigid-instances the result. Pin a path
explicitly to compare:

```text
http://localhost:5173/supermesh-scale?renderer=babylonjs&model=nm-m-supermesh&path=hybrid&mode=explore
```

`path=supermesh` is the maximal-supermesh baseline: every actor skins all 8,532
vertices and hidden modules are clipped only after skinning.

The manifest stores six common body-part slots and each slot's available module
variations. It stores no permutation table. The GUI discovers that catalog and
maps it onto six Shado published fields statically defined for this exact NM_M
actor schema. Its 3,125 possible combinations are composed only when actors are
spawned. The target selector edits one live actor; actors can be cloned,
randomized, composed deterministically, removed, or resized as a population.
The current target is restored by the URL fields `count`, `target`, `clip`,
`bd`, `hn`, `pt`, `lg`, `hr`, and `ey`.

The route loads the complete NM_M translation: all 329 clips and 17,381 frames.
The `clip` query accepts every runtime name in the published index. The atlas is
delivered as a gzip binary plus gzip extras index rather than base64 JSON.

Set `mode=benchmark` to add actors progressively, measure steady frame and GPU
times, stop after two levels exceed 50 ms p95, and expose the JSON result as
`window.__shadoSupermeshScale`.
The development server also saves the completed result to
`benchmark-results/supermesh-latest.json`.

Useful query parameters:

- `path=bat|bat-thin|supermesh|hybrid|cached` (`bat*` paths are local captured-implementation benchmarks)
- `quality=full|medium|low|rigid`
- `counts=1,10,25,50,100,200,400,800`
- `warmupMs=800` and `sampleMs=3000` — each level spins for a wall-clock window
  and samples whatever the host delivers, rather than waiting on a frame quota
  that a throttled browser never meets

Rows carry `fps` and `frames` alongside the timings so a starved window is
visible rather than silent. Two verdicts mean "do not read the frame columns":
`no samples` (the host delivered no frames at all) and `host-paced` (frame time
tracked the browser's rAF cadence, not render cost — detected when mean frame
time exceeds 100 ms and dwarfs measured GPU time). Neither contributes to the
60/30 fps limits or the peak-throughput figure. The GPU column stays meaningful
in both cases, because it measures the work inside a frame rather than how
often frames are issued.

## Vercel

Vercel reads `vercel.json` from the project's Root Directory and ignores any
other copy, so which file is live depends on how the project is configured:

- Root Directory `sandbox` — this directory's `vercel.json` applies. Keep
  **Include source files outside of the Root Directory** enabled so the linked
  parent `@knervous/shado` package is available during install.
- Root Directory `.` (the repository root) — `../vercel.json` applies instead.
  It carries only the rewrites and headers; the build command and output
  directory come from the project settings (`sandbox/dist`).

Either way the config restores SPA deep links (`/world`, `/world-editor`,
`/supermesh-scale` are client routes with no file on disk, so without the
rewrite they 404 while `/?renderer=…` still works) and supplies the
cross-origin-isolation headers Shado's shared-memory paths require. **Keep the
`rewrites` and `headers` blocks in the two files identical.**
The exact-model share URLs are:

```text
/?renderer=babylonjs&model=nm-m-supermesh&mode=explore
/hum-wardrobe?renderer=babylonjs&model=hum
```

Both are plain client routes with no file on disk, so they depend on the
catch-all rewrite above; the wardrobe bundles under `public/shado/{hum,huf}/`
are tracked and ship with the build.

The non-rendering residency benchmark can be run separately:

```bash
npm run benchmark:supermesh-residency -- --max=1000000
```
