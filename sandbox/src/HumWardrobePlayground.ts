import * as BABYLON from '@babylonjs/core';
import { SceneInstrumentation } from '@babylonjs/core/Instrumentation/sceneInstrumentation';
import {
  ShadoActor,
  ShadoInstanceContainer,
  ShadoModuleDrawSet,
  field,
  gpuStruct,
  splitMeshesIntoModules,
} from '@knervous/shado';
import { createHumWardrobeMaterial } from './HumWardrobeMaterial';
import { createHumWardrobeUi, type HumWardrobeControls } from './HumWardrobeUi';

/**
 * The Ryzom-derived `hum` wardrobe, drawn as per-variant modules.
 *
 * The asset is the promoted Fyros male body: 74 submeshes grouped into 35
 * (piece, variation) outfits across 7 body pieces — four civilian wardrobes,
 * three armour tiers and seven hairstyles. Vendored by
 * `client/scripts/vendor-hum-wardrobe-demo.mjs` from the post-map `.babylon`,
 * because only that artifact carries the {piece, variation, texNum} stamps the
 * wardrobe map produces.
 *
 * What this page is for: showing that instance count and draw count are
 * independent under the module pattern. Scale to 10,000 actors and the draw
 * count stays at one per populated module, because each module submits its own
 * compact actor list rather than the geometry being re-submitted per instance.
 */

/** `?model=huf` swaps the whole bundle; both are vendored side by side. */
const MODELS = ['hum', 'huf'] as const;
const MODEL = (() => {
  const requested = new URLSearchParams(window.location.search).get('model');
  return (MODELS as readonly string[]).includes(requested ?? '') ? requested! : 'hum';
})();
const ASSET_ROOT = `/shado/${MODEL}`;

@gpuStruct({ name: 'HumWardrobeActor' })
class HumWardrobeActor extends ShadoActor {}

@gpuStruct({ name: 'HumWardrobeContainer' })
class HumWardrobeContainer extends ShadoInstanceContainer<HumWardrobeActor> {
  @field({ arrayOf: 'vec4' }) appearance!: Float32Array;

  public ensureAppearance(count: number): void {
    if (this.getVarArrayCount('appearance') >= count) return;
    this.resizeVarArray('appearance', count);
  }

  public writeAppearance(index: number, value: ArrayLike<number>): void {
    this.ensureAppearance(index + 1);
    this.writeVarArrayRange('appearance', index, value);
  }
}

type WardrobeManifest = {
  model: string;
  runtimeScale: number;
  runtimeYawCorrection: number;
  boundingBox: { min: number[]; max: number[] } | null;
  submeshCount: number;
  submeshes: {
    name: string;
    piece: string;
    variation: string;
    texNum: string;
    atlasIndex: number;
  }[];
  pieces: { piece: string; variations: { variation: string; submeshOrdinals: number[] }[] }[];
  atlas: { file: string; side: number; columns: number; rows: number; layers: string[] };
  response?: {
    file: string;
    side: number;
    columns: number;
    rows: number;
    maskedLayers: number;
    emblazonedLayers: number;
  } | null;
  appearance?: {
    complexions: [number, number, number][];
    hidePiece: string;
    hair: { base: number; count: number };
    face: { base: number; count: number };
    uniforms: {
      uniform: string;
      label: string;
      tier: number;
      texture: number;
      palette: { neutralize: number; pieces: Record<string, [number, number, number]> } | null;
      unmintedPalette: { neutralize: number; pieces: Record<string, [number, number, number]> } | null;
      device: { strength: number; colour: [number, number, number] } | null;
      meshes: Record<string, string> | null;
      minted: boolean;
      layers: { piece: string; layers: { texNum: string; atlasIndex: number; layer: string }[] }[];
    }[];
    faces: { variation: string; layers: { texNum: string; atlasIndex: number; layer: string }[] }[];
  } | null;
  vat: { bin: string; fps: number; animations: { name: string; from: number; to: number }[] };
  geometry: { file: string };
};

const PIECE_LABELS: Record<string, string> = {
  ch: 'Chest',
  lg: 'Legs',
  ft: 'Feet',
  ua: 'Arms',
  hn: 'Hands',
  he: 'Face',
  hr: 'Hair',
};

async function fetchGunzip(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const raw = new Uint8Array(await response.arrayBuffer());
  // Dev servers commonly serve a .gz with `Content-Encoding: gzip`, in which
  // case the browser has already decompressed it and gunzipping again throws.
  // Sniff the magic instead of trusting the extension.
  if (raw[0] !== 0x1f || raw[1] !== 0x8b) return raw;
  const stream = new Blob([raw as BlobPart])
    .stream()
    .pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * Decodes the vendored PNG grid sheet into a 2D array texture, one layer per
 * outfit. A grid sheet rather than a compressed container keeps the sandbox
 * free of a transcoder while staying under 4 MB for 43 layers.
 */
async function loadAtlasArray(
  scene: BABYLON.Scene,
  sheet: { side: number; columns: number; file: string },
  layerCount: number,
  name: string
): Promise<BABYLON.RawTexture2DArray> {
  const { side, columns, file } = sheet;
  const layers = { length: layerCount };
  // createImageBitmap rather than an <img> + decode(): image decoding is tied
  // to the rendering path, so a throttled or backgrounded page can leave
  // `decode()` pending indefinitely. This route does not depend on painting.
  const response = await fetch(`${ASSET_ROOT}/${file}`);
  if (!response.ok) throw new Error(`${file}: ${response.status}`);
  const image = await createImageBitmap(await response.blob());

  const canvas = document.createElement('canvas');
  canvas.width = side;
  canvas.height = side;
  const context = canvas.getContext('2d', { willReadFrequently: true })!;
  const data = new Uint8Array(layers.length * side * side * 4);
  for (let layer = 0; layer < layers.length; layer++) {
    context.clearRect(0, 0, side, side);
    context.drawImage(
      image,
      (layer % columns) * side,
      Math.floor(layer / columns) * side,
      side,
      side,
      0,
      0,
      side,
      side
    );
    data.set(context.getImageData(0, 0, side, side).data, layer * side * side * 4);
  }
  image.close();

  // No mip chain. WebGPU's RawTexture2DArray mip generation fails for deep
  // arrays - every level past 0 comes back black - so a wardrobe atlas with one
  // layer per submesh renders black the moment a surface is far enough to
  // sample a mip. That reads as "most of the crowd is black except the nearest
  // bodies", which is exactly what it looks like. Shallow arrays are fine, so
  // the cutoff mirrors the game client's.
  const useMipChain = layers.length <= 8;
  const texture = new BABYLON.RawTexture2DArray(
    data,
    side,
    side,
    layers.length,
    BABYLON.Constants.TEXTUREFORMAT_RGBA,
    scene,
    useMipChain,
    false,
    useMipChain
      ? BABYLON.Constants.TEXTURE_TRILINEAR_SAMPLINGMODE
      : BABYLON.Constants.TEXTURE_LINEAR_LINEAR
  );
  texture.name = name;
  return texture;
}

export class HumWardrobePlayground {
  public static async CreateScene(
    engine: BABYLON.Engine,
    canvas: HTMLCanvasElement
  ): Promise<BABYLON.Scene> {
    const scene = new BABYLON.Scene(engine);
    scene.clearColor = BABYLON.Color4.FromHexString('#0b1017ff');
    (globalThis as any).__shadoScene = scene;

    const instrumentation = new SceneInstrumentation(scene);
    instrumentation.captureFrameTime = true;

    const camera = new BABYLON.ArcRotateCamera(
      'hum-wardrobe-camera',
      -Math.PI / 2,
      1.15,
      34,
      new BABYLON.Vector3(0, 3, 0),
      scene
    );
    camera.attachControl(canvas, true);
    camera.lowerRadiusLimit = 6;
    camera.upperRadiusLimit = 900;
    camera.wheelPrecision = 12;
    camera.panningSensibility = 40;

    const lightDirection = new BABYLON.Vector3(0.45, 0.82, 0.35).normalize();

    const manifest: WardrobeManifest = await (
      await fetch(`${ASSET_ROOT}/${MODEL}.wardrobe.json`)
    ).json();

    // ── geometry ───────────────────────────────────────────────────────────
    // A blob URL rather than `data:` - the parsed scene is ~15 MB of JSON, and
    // a data URL of that size costs seconds before the loader even starts.
    const sceneBlob = new Blob(
      [await fetchGunzip(`${ASSET_ROOT}/${manifest.geometry.file}`) as BlobPart],
      { type: 'application/babylon' }
    );
    const sceneUrl = URL.createObjectURL(sceneBlob);
    let imported: BABYLON.AssetContainer;
    try {
      imported = await BABYLON.LoadAssetContainerAsync(sceneUrl, scene, {
        pluginExtension: '.babylon',
      });
    } finally {
      URL.revokeObjectURL(sceneUrl);
    }
    imported.addAllToScene();

    const byName = new Map<string, BABYLON.Mesh>();
    for (const mesh of imported.meshes) {
      if (mesh instanceof BABYLON.Mesh && mesh.getTotalVertices() > 0) {
        byName.set(mesh.name, mesh);
      }
    }
    const skeleton = imported.skeletons[0] ?? null;

    // Source meshes in manifest order, each stamped with the per-vertex pair
    // the shader reads: (atlas layer, global submesh ordinal). Ordinals stay
    // global across modules so the appearance array indexing never moves.
    const sources: BABYLON.Mesh[] = [];
    for (const [ordinal, submesh] of manifest.submeshes.entries()) {
      const mesh = byName.get(submesh.name);
      if (!mesh) throw new Error(`manifest names ${submesh.name}, which the geometry lacks`);
      mesh.computeWorldMatrix(true);
      mesh.bakeTransformIntoVertices(mesh.getWorldMatrix());
      const count = mesh.getTotalVertices();
      const stamp = new Float32Array(count * 2);
      for (let vertex = 0; vertex < count; vertex++) {
        stamp[vertex * 2] = submesh.atlasIndex;
        stamp[vertex * 2 + 1] = ordinal;
      }
      mesh.setVerticesData('submeshData', stamp, false, 2);
      mesh.material?.dispose();
      sources.push(mesh);
    }

    // ── textures ───────────────────────────────────────────────────────────
    const atlas = await loadAtlasArray(
      scene,
      manifest.atlas,
      manifest.atlas.layers.length,
      `${MODEL}-wardrobe-atlas`,
    );
    // The mask that turns one albedo layer into an outfit, a complexion and a
    // banner. Optional: a bundle vendored before the response sheet existed
    // still runs, it just cannot protect skin from a dye.
    const response = manifest.response
      ? await loadAtlasArray(
          scene,
          manifest.response,
          manifest.atlas.layers.length,
          `${MODEL}-wardrobe-response`,
        )
      : null;

    // ── the role vocabulary, as the manifest carries it ───────────────────
    const appearanceData = manifest.appearance ?? null;
    const COMPLEXIONS: [number, number, number][] = appearanceData?.complexions ?? [[1, 1, 1]];
    const HIDE_PIECE = appearanceData?.hidePiece ?? "--";
    const UNIFORMS = appearanceData?.uniforms ?? [];
    const FACES = appearanceData?.faces ?? [];

    /** `uniform -> piece -> texNum -> atlasIndex`, for the layer redirect. */
    const uniformLayerIndex = new Map<string, Map<string, Map<string, number>>>();
    for (const uniform of UNIFORMS) {
      const byPiece = new Map<string, Map<string, number>>();
      for (const group of uniform.layers) {
        byPiece.set(
          group.piece,
          new Map(group.layers.map((layer) => [layer.texNum, layer.atlasIndex])),
        );
      }
      uniformLayerIndex.set(uniform.uniform, byPiece);
    }
    /** `faceVariation -> texNum -> atlasIndex`. */
    const faceLayerIndex = new Map(
      FACES.map((face) => [
        face.variation,
        new Map(face.layers.map((layer) => [layer.texNum, layer.atlasIndex])),
      ]),
    );
    const vatBytes = await fetchGunzip(`${ASSET_ROOT}/${manifest.vat.bin}`);
    const baker = new BABYLON.VertexAnimationBaker(scene, skeleton as any);
    const vatTexture = baker.textureFromBakedVertexData(
      new Uint16Array(vatBytes.buffer, vatBytes.byteOffset, vatBytes.byteLength / 2) as any
    );
    vatTexture.name = `${manifest.model}-wardrobe-vat`;

    // ── arena ──────────────────────────────────────────────────────────────
    const ok = await HumWardrobeContainer.initialize(engine, {
      backend: (engine as any).isWebGPU ? 'storage' : 'datatex',
      extra: HumWardrobeActor,
      // No AssemblyScript reducer here: the demo drives visibility itself, and
      // runtime compilation would pull binaryen into the sandbox bundle.
      wasm: false,
    });
    if (!ok) throw new Error('Unable to initialize the Shado arena');
    const container = new HumWardrobeContainer(engine);
    container.markArenaDirty();
    container.commit();

    // ── modules ────────────────────────────────────────────────────────────
    const geometry = splitMeshesIntoModules(sources, {
      groupKey: (_mesh, index) =>
        `${manifest.submeshes[index].piece}:${manifest.submeshes[index].variation}`,
      preserveAttributes: [{ kind: 'submeshData', stride: 2 }],
      name: (key) => `${MODEL}#${key}`,
    });
    const draws = new ShadoModuleDrawSet(engine, geometry);
    draws.registerThinInstanceAttribute('matrix', 16);

    const moduleInfo = geometry.map((module) => {
      const first = manifest.submeshes[module.sourceIndices[0]];
      return { key: module.key, piece: first.piece, variation: first.variation };
    });

    let clock = 0;
    const materials: BABYLON.ShaderMaterial[] = [];
    for (const [index, module] of draws.modules.entries()) {
      const mesh = module.mesh as unknown as BABYLON.Mesh;
      mesh.skeleton = skeleton;
      mesh.alwaysSelectAsActiveMesh = true;
      mesh.isPickable = false;
      mesh.position.setAll(0);
      mesh.rotation.setAll(0);
      mesh.scaling.setAll(1);
      const material = createHumWardrobeMaterial(scene, `${MODEL}Wardrobe#${module.key}`, {
        container,
        response,
        atlas,
        vatTexture,
        submeshCount: manifest.submeshCount,
        time: () => clock,
        lightDirection,
        bindSelection: (target) => draws.bindSelection(index, target),
      });
      module.selection?.bind(material);
      mesh.material = material;
      materials.push(material);
    }

    // ── actors ─────────────────────────────────────────────────────────────
    const pieces = manifest.pieces;
    const pieceOf = new Map<string, number>(pieces.map((piece, index) => [piece.piece, index]));
    /** Per actor, the chosen variation index for each piece. */
    let worn: Uint8Array = new Uint8Array(0);
    /** Per actor per piece, an RGB tint. */
    let tints: Float32Array = new Float32Array(0);
    let tintsRandomized = false;
    let tintSeed = 5;
    let actorCount = 0;
    const clips = manifest.vat.animations;
    const idle = clips.find((clip) => clip.name === 'p01') ?? clips[0];
    // 72 EQ codes, but only ~30 are distinct bakes - the rest alias onto idle,
    // so drawing a random *code* would put most of the crowd in the same pose.
    // Dedupe by frame range to get the clips that actually differ.
    const distinctClips = clips.filter(
      (clip, index) =>
        clips.findIndex((other) => other.from === clip.from && other.to === clip.to) ===
        index
    );
    /** Per actor, the clip it is playing. null means "follow the crowd clip". */
    let actorClips: number[] = [];
    /** Index into distinctClips, or -1 for "each actor picks its own". */
    let crowdClip = -1;

    const rng = (seed: number) => {
      let state = seed >>> 0;
      return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 4294967296;
      };
    };

    // Two rows past the submeshes carry what belongs to the person rather than
    // to a garment: complexion, then heraldry. Same stride the game uses.
    const ENTITY_ROWS = 2;
    const APPEARANCE_STRIDE = manifest.submeshCount + ENTITY_ROWS;

    /** How the crowd is dressed. '' none, 'mixed' one per actor, else a code. */
    let uniformMode = '';
    /** '' keeps the body's own visage, 'auto' spreads the minted eight. */
    let faceMode = 'auto';
    /** 'auto' spreads the eight complexions; a digit pins one. */
    let complexionMode = 'auto';
    /** The crowd's own outfit roll, kept so turning a uniform off restores it. */
    let wornBase = new Uint8Array(0);

    /** Which uniform an actor wears, or -1 for none. Indexes `UNIFORMS`. */
    let actorUniform = new Int16Array(0);
    /** Which minted face, as an index into `FACES`; -1 keeps the body's own. */
    let actorFace = new Int16Array(0);
    /** Which complexion multiplier, indexing `COMPLEXIONS`. */
    let actorComplexion = new Uint8Array(0);
    /** Pieces a uniform declines outright, so the module filter can skip them. */
    let hiddenPieces = new Uint8Array(0);

    /**
     * Writes one actor: the variation it wears visible, the rest at -1, then
     * the two entity rows.
     *
     * Three things can move a submesh's atlas layer, in the order the game
     * resolves them: a minted face wins for the head, a role uniform's repaint
     * wins for anything else, and otherwise the mesh keeps its own art. That
     * order is the point — a face individuates a person and a uniform
     * individuates a faction, so two guards of one watch should not share a jaw.
     */
    const writeActorAppearance = (actorIndex: number) => {
      const base = actorIndex * APPEARANCE_STRIDE;
      const uniformIndex = actorUniform[actorIndex] ?? -1;
      const uniform = uniformIndex >= 0 ? UNIFORMS[uniformIndex] : null;
      const uniformLayers = uniform ? uniformLayerIndex.get(uniform.uniform) : undefined;
      const faceIndex = actorFace[actorIndex] ?? -1;
      const faceLayers =
        faceIndex >= 0 ? faceLayerIndex.get(FACES[faceIndex].variation) : undefined;
      // A uniform whose art was minted for this body keeps the painted sheet;
      // one that was not falls back to the dye equivalent of it.
      const dye = uniform
        ? (uniform.minted ? uniform.palette : (uniform.palette ?? uniform.unmintedPalette))
        : null;

      for (const [pieceIndex, piece] of pieces.entries()) {
        const chosen = worn[actorIndex * pieces.length + pieceIndex];
        const hidden = hiddenPieces[actorIndex * pieces.length + pieceIndex] === 1;
        const tintBase = (actorIndex * pieces.length + pieceIndex) * 3;
        const paletteDye = dye?.pieces[piece.piece];
        const r = paletteDye ? paletteDye[0] : tints[tintBase];
        const g = paletteDye ? paletteDye[1] : tints[tintBase + 1];
        const b = paletteDye ? paletteDye[2] : tints[tintBase + 2];
        for (const [variationIndex, variation] of piece.variations.entries()) {
          const visible = variationIndex === chosen && !hidden;
          for (const ordinal of variation.submeshOrdinals) {
            const submesh = manifest.submeshes[ordinal];
            let slice = submesh.atlasIndex;
            if (piece.piece === 'he' && faceLayers?.has(submesh.texNum)) {
              slice = faceLayers.get(submesh.texNum)!;
            } else if (uniformLayers?.get(piece.piece)?.has(submesh.texNum)) {
              slice = uniformLayers.get(piece.piece)!.get(submesh.texNum)!;
            }
            container.writeAppearance(base + ordinal, [visible ? slice : -1, r, g, b]);
          }
        }
      }

      const complexion = COMPLEXIONS[actorComplexion[actorIndex] % COMPLEXIONS.length] ?? [1, 1, 1];
      container.writeAppearance(base + manifest.submeshCount, [
        dye?.neutralize ?? 0,
        complexion[0],
        complexion[1],
        complexion[2],
      ]);
      const device = uniform?.device ?? null;
      container.writeAppearance(base + manifest.submeshCount + 1, [
        device?.strength ?? 0,
        device?.colour[0] ?? 1,
        device?.colour[1] ?? 1,
        device?.colour[2] ?? 1,
      ]);
    };

    /**
     * Resolves the three appearance modes down to per-actor choices.
     *
     * A uniform may compose the wardrobe per piece rather than take one tier for
     * the whole body — a trader is a civilian coat over a robe skirt — and it may
     * decline a piece outright, which the ladder cannot express because it
     * degrades to the nearest lower variation and never to nothing. So `worn` is
     * rebuilt from `wornBase` here rather than mutated in place, and the module
     * filter reads the same arrays, or the draw list and the appearance plane
     * would disagree about who is wearing what.
     */
    /** The ladder's own rule: highest variation not above the one asked for. */
    const ladderChoice = (piece: (typeof pieces)[number], requested: number): number => {
      let chosen = 0;
      for (const [index, variation] of piece.variations.entries()) {
        if (Number(variation.variation) <= requested) chosen = index;
      }
      return chosen;
    };

    const applyAppearanceModes = () => {
      const spread = rng(9001);
      for (let actor = 0; actor < actorCount; actor += 1) {
        const uniformIndex =
          uniformMode === ''
            ? -1
            : uniformMode === 'mixed'
              ? Math.floor(spread() * UNIFORMS.length)
              : UNIFORMS.findIndex((entry) => entry.uniform === uniformMode);
        actorUniform[actor] = uniformIndex;
        actorFace[actor] =
          faceMode === ''
            ? -1
            : faceMode === 'auto'
              ? FACES.length
                ? Math.floor(spread() * FACES.length)
                : -1
              : FACES.findIndex((face) => face.variation === faceMode);
        actorComplexion[actor] =
          complexionMode === 'auto'
            ? Math.floor(spread() * COMPLEXIONS.length)
            : Number(complexionMode) % COMPLEXIONS.length;

        const entry = uniformIndex >= 0 ? UNIFORMS[uniformIndex] : null;
        const composition = entry?.meshes ?? null;
        for (const [pieceIndex, piece] of pieces.entries()) {
          const slot = actor * pieces.length + pieceIndex;
          const composed = composition?.[piece.piece] ?? null;
          if (composed === HIDE_PIECE) {
            hiddenPieces[slot] = 1;
            worn[slot] = wornBase[slot];
            continue;
          }
          hiddenPieces[slot] = 0;
          if (composed === null) {
            // A uniform with minted art has to pin the tier its layers were
            // painted from. The repaint reuses the sheet a mesh already
            // samples, so tier-2 chain paint on a tier-0 tunic is not a
            // recolour, it is the wrong UVs — the seams land mid-torso. Only
            // the crowd's own roll is free to vary, and only when no uniform
            // is claiming the piece.
            worn[slot] = entry?.minted ? ladderChoice(piece, entry.tier) : wornBase[slot];
            continue;
          }
          // A body missing that graft degrades instead of vanishing.
          worn[slot] = ladderChoice(piece, Number(composed));
        }
      }
      for (let actor = 0; actor < actorCount; actor += 1) writeActorAppearance(actor);
      container.commit();
    };

    const populate = (count: number, variability: number, seed = 1) => {
      const random = rng(seed);
      const previous = actorCount;
      actorCount = count;
      worn = new Uint8Array(count * pieces.length);
      wornBase = new Uint8Array(count * pieces.length);
      hiddenPieces = new Uint8Array(count * pieces.length);
      actorUniform = new Int16Array(count).fill(-1);
      actorFace = new Int16Array(count).fill(-1);
      actorComplexion = new Uint8Array(count);
      actorClips = new Array(count).fill(0);
      tints = new Float32Array(count * pieces.length * 3).fill(1);
      // Re-rolled after the loop below when the crowd was dyed, so growing the
      // crowd does not silently reset every tint to white.

      for (let actor = 0; actor < count; actor++) {
        for (const [pieceIndex, piece] of pieces.entries()) {
          // variability 0 dresses the whole crowd identically, which is the
          // useful control: identical outfits collapse to the fewest modules.
          wornBase[actor * pieces.length + pieceIndex] =
            random() < variability ? Math.floor(random() * piece.variations.length) : 0;
        }
      }
      worn.set(wornBase);
      // Reserve the arena for the whole crowd BEFORE touching it: both the
      // actor records and the appearance plane, which is submeshCount vec4s per
      // actor and by far the larger of the two.
      //
      // Growing it implicitly instead silently truncates. The appearance
      // resize is capped by whatever the arena can still fit, so the tail of
      // the plane never gets allocated and reads back as zeroes - slice 0 with
      // tint (0, 0, 0). The submesh still draws, because slice 0 is a real
      // atlas layer; it just draws pure black. That is the "most actors have
      // black submeshes" symptom, and it gets worse the bigger the crowd.
      container.reserveInstances(count);
      container.reserveVarArray('appearance', count * APPEARANCE_STRIDE);
      container.ensureAppearance(count * APPEARANCE_STRIDE);
      for (let actor = previous; actor < count; actor++) {
        container.addInstance(true);
      }
      const columns = Math.ceil(Math.sqrt(count));
      const spacing = 4.2;
      for (let actor = 0; actor < count; actor++) {
        const record = container.children[actor];
        const x = (actor % columns) - columns / 2;
        const z = Math.floor(actor / columns) - columns / 2;
        record.translation = new Float32Array([x * spacing, 0, z * spacing, 1]);
        record.rotation = new Float32Array([0, 0, 0, 1]);
        // Every actor animates independently: its own clip and its own phase
        // within it. The arena stores a clip range per actor, so this costs
        // four floats each and no extra draws - the modules a crowd of mixed
        // poses needs are the same modules a synchronised crowd needs.
        const clip =
          crowdClip >= 0
            ? distinctClips[crowdClip]
            : distinctClips[Math.floor(random() * distinctClips.length)];
        actorClips[actor] = clips.indexOf(clip);
        record.animationBuffer = new Float32Array([
          clip.from,
          clip.to,
          random() * (clip.to - clip.from),
          manifest.vat.fps,
        ]);
        record.visibleFlag = 1;
        record.visibleIndex = actor;
        writeActorAppearance(actor);
      }
      container.applyVisibilityReduction(
        Uint32Array.from({ length: count }, (_value, index) => index)
      );
      container.commit();
      // populate() rebuilt the tint plane as white; restore the dye if the
      // crowd was randomized, so resizing does not silently undo it.
      applyAppearanceModes();
      if (tintsRandomized) randomizeTints(tintSeed);
      // Thin instances are the draw-count adapter; the shader reads transforms
      // from the arena, so the matrix itself is never sampled.
      for (let actor = previous; actor < count; actor++) {
        draws.addThinInstance(BABYLON.Matrix.Identity());
      }
    };

    /**
     * Distinct dye per actor per piece.
     *
     * Drawn in HSL rather than raw RGB: uniform random channels average to
     * muddy grey and swamp the albedo underneath, whereas a random hue at
     * controlled saturation reads as dyed cloth and still lets the texture
     * through. Costs nothing at the draw level - tint rides in the appearance
     * vec4 that is already being written.
     */
    const randomizeTints = (seed = 5) => {
      const random = rng(seed);
      for (let actor = 0; actor < actorCount; actor++) {
        for (let pieceIndex = 0; pieceIndex < pieces.length; pieceIndex++) {
          const base = (actor * pieces.length + pieceIndex) * 3;
          const color = BABYLON.Color3.FromHSV(
            random() * 360,
            0.35 + random() * 0.4,
            0.75 + random() * 0.25,
          );
          tints[base] = color.r;
          tints[base + 1] = color.g;
          tints[base + 2] = color.b;
        }
        writeActorAppearance(actor);
      }
      container.commit();
    };

    const resetTints = () => {
      tints.fill(1);
      for (let actor = 0; actor < actorCount; actor++) writeActorAppearance(actor);
      container.commit();
    };

    const applyPieceTint = (piece: string, color: BABYLON.Color3) => {
      const pieceIndex = pieceOf.get(piece);
      if (pieceIndex === undefined) return;
      for (let actor = 0; actor < actorCount; actor++) {
        const base = (actor * pieces.length + pieceIndex) * 3;
        tints[base] = color.r;
        tints[base + 1] = color.g;
        tints[base + 2] = color.b;
        writeActorAppearance(actor);
      }
      container.commit();
    };

    const setPieceVariation = (piece: string, variationIndex: number | null) => {
      const pieceIndex = pieceOf.get(piece);
      if (pieceIndex === undefined) return;
      const entry = pieces[pieceIndex];
      const random = rng(7 + pieceIndex);
      for (let actor = 0; actor < actorCount; actor++) {
        worn[actor * pieces.length + pieceIndex] =
          variationIndex === null
            ? Math.floor(random() * entry.variations.length)
            : variationIndex;
        writeActorAppearance(actor);
      }
      container.commit();
    };

    const setClip = (name: string) => {
      const random = rng(31);
      crowdClip =
        name === 'mixed'
          ? -1
          : Math.max(
              0,
              distinctClips.findIndex((clip) => clip.name === name),
            );
      for (let actor = 0; actor < actorCount; actor++) {
        const clip =
          crowdClip >= 0
            ? distinctClips[crowdClip]
            : distinctClips[Math.floor(random() * distinctClips.length)];
        actorClips[actor] = clips.indexOf(clip);
        container.children[actor].animationBuffer = new Float32Array([
          clip.from,
          clip.to,
          // A distinct phase as well as a distinct clip: a hundred actors
          // starting the same walk on the same frame still reads as one puppet.
          random() * (clip.to - clip.from),
          manifest.vat.fps,
        ]);
      }
      container.commit();
    };

    const controls: HumWardrobeControls = {
      model: MODEL,
      models: [...MODELS],
      submeshCount: manifest.submeshCount,
      moduleCount: geometry.length,
      pieceCount: pieces.length,
      setModel: (next) => {
        const url = new URL(window.location.href);
        url.searchParams.set('model', next);
        window.location.assign(url.toString());
      },
      pieces: pieces.map((piece) => ({
        piece: piece.piece,
        label: PIECE_LABELS[piece.piece] ?? piece.piece,
        variations: piece.variations.map((variation) => variation.variation),
      })),
      clips: ['mixed', ...distinctClips.map((clip) => clip.name)],
      uniforms: UNIFORMS.map((entry) => ({
        uniform: entry.uniform,
        label: entry.label,
        texture: entry.texture,
        minted: entry.minted,
        dyed: Boolean(entry.palette ?? entry.unmintedPalette),
        device: Boolean(entry.device?.strength),
      })),
      faces: FACES.map((face) => face.variation),
      complexions: COMPLEXIONS.length,
      hasResponse: Boolean(response),
      setUniform: (mode) => {
        uniformMode = mode;
        applyAppearanceModes();
      },
      setFace: (mode) => {
        faceMode = mode;
        applyAppearanceModes();
      },
      setComplexion: (mode) => {
        complexionMode = mode;
        applyAppearanceModes();
      },
      setCount: (count, variability) => populate(count, variability),
      setVariability: (variability) => populate(actorCount, variability, 2),
      setPieceVariation,
      setPieceTint: applyPieceTint,
      randomizeTints: () => {
        tintsRandomized = true;
        randomizeTints(tintSeed++);
      },
      resetTints: () => {
        tintsRandomized = false;
        resetTints();
      },
      setClip,
    };
    const ui = createHumWardrobeUi(controls);

    populate(256, 0.65);

    scene.onBeforeRenderObservable.add(() => {
      clock += engine.getDeltaTime() / 1000;
      draws.refresh(container.visibleActorIndices, (actorIndex, moduleIndex) => {
        const info = moduleInfo[moduleIndex];
        const pieceIndex = pieceOf.get(info.piece)!;
        const slot = actorIndex * pieces.length + pieceIndex;
        if (hiddenPieces[slot] === 1) return false;
        const chosen = worn[slot];
        return pieces[pieceIndex].variations[chosen].variation === info.variation;
      });
    });

    let lastReport = 0;
    scene.onAfterRenderObservable.add(() => {
      const now = performance.now();
      if (now - lastReport < 250) return;
      lastReport = now;
      const stats = draws.lastStats;
      ui.report({
        actors: actorCount,
        drawCalls: instrumentation.drawCallsCounter.current,
        moduleDraws: stats.populatedModules,
        moduleTotal: stats.moduleCount,
        submittedVertices: stats.submittedVertices,
        baselineVertices: stats.baselineVertices,
        vertexWorkReduction: stats.vertexWorkReduction,
        fps: engine.getFps(),
        frameMs: instrumentation.frameTimeCounter.lastSecAverage,
      });
    });

    scene.onDisposeObservable.add(() => {
      ui.dispose();
      instrumentation.dispose();
      for (const material of materials) material.dispose(true, false);
      draws.dispose();
      atlas.dispose();
      vatTexture.dispose();
      container.dispose();
    });

    (globalThis as any).__humWardrobe = {
      draws,
      container,
      manifest,
      /** Counts visible submeshes whose tint read back as zero (i.e. black). */
      auditAppearance() {
        const arena = (container as any).arena ?? (container as any)._arena;
        // The segment's float offset, not getVarArrayPtr - that returns a
        // backing-specific pointer (0 here) and reading from it lands one vec4
        // early, which makes element 0 look corrupt when it is fine.
        const base = (container as any)._varSeg.appearance.offF as number;
        const plane = arena.f32.subarray(base) as Float32Array;
        let visible = 0;
        let blackTint = 0;
        for (let actor = 0; actor < actorCount; actor++) {
          for (let ordinal = 0; ordinal < manifest.submeshCount; ordinal++) {
            const base = (actor * manifest.submeshCount + ordinal) * 4;
            if (plane[base] < 0) continue;
            visible++;
            if (plane[base + 1] === 0 && plane[base + 2] === 0 && plane[base + 3] === 0) {
              blackTint++;
            }
          }
        }
        return {
          visible,
          blackTint,
          varArrayCount: container.getVarArrayCount('appearance'),
          needed: actorCount * manifest.submeshCount,
        };
      },
    };
    return scene;
  }
}
