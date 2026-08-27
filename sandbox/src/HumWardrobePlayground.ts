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
 * The asset is the promoted Fyros male body: 43 submeshes grouped into 17
 * (piece, variation) outfits across 7 body pieces. Vendored by
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
  manifest: WardrobeManifest
): Promise<BABYLON.RawTexture2DArray> {
  const { side, columns, layers, file } = manifest.atlas;
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
  texture.name = 'hum-wardrobe-atlas';
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
    const atlas = await loadAtlasArray(scene, manifest);
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

    /** Writes one actor's appearance: chosen variation visible, rest at -1. */
    const writeActorAppearance = (actorIndex: number) => {
      const base = actorIndex * manifest.submeshCount;
      for (const [pieceIndex, piece] of pieces.entries()) {
        const chosen = worn[actorIndex * pieces.length + pieceIndex];
        const tintBase = (actorIndex * pieces.length + pieceIndex) * 3;
        for (const [variationIndex, variation] of piece.variations.entries()) {
          const visible = variationIndex === chosen;
          for (const ordinal of variation.submeshOrdinals) {
            container.writeAppearance(base + ordinal, [
              visible ? manifest.submeshes[ordinal].atlasIndex : -1,
              tints[tintBase],
              tints[tintBase + 1],
              tints[tintBase + 2],
            ]);
          }
        }
      }
    };

    const populate = (count: number, variability: number, seed = 1) => {
      const random = rng(seed);
      const previous = actorCount;
      actorCount = count;
      worn = new Uint8Array(count * pieces.length);
      actorClips = new Array(count).fill(0);
      tints = new Float32Array(count * pieces.length * 3).fill(1);
      // Re-rolled after the loop below when the crowd was dyed, so growing the
      // crowd does not silently reset every tint to white.

      for (let actor = 0; actor < count; actor++) {
        for (const [pieceIndex, piece] of pieces.entries()) {
          // variability 0 dresses the whole crowd identically, which is the
          // useful control: identical outfits collapse to the fewest modules.
          worn[actor * pieces.length + pieceIndex] =
            random() < variability ? Math.floor(random() * piece.variations.length) : 0;
        }
      }
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
      container.reserveVarArray('appearance', count * manifest.submeshCount);
      container.ensureAppearance(count * manifest.submeshCount);
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
        const chosen = worn[actorIndex * pieces.length + pieceIndex];
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
