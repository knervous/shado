import * as BABYLON from '@babylonjs/core';
import '@babylonjs/loaders/glTF';
import { EngineInstrumentation } from '@babylonjs/core/Instrumentation/engineInstrumentation';
import {
  createShadoVatShowcase,
  createShadoVatShowcaseUi,
  createShadoShowcaseEnvironment,
  type ShadoVatShowcaseModel,
  type ShadoVatShowcaseStats,
} from '@knervous/shado';

/** One row of public/shado/ryzom/ryzom-catalog.json, written by sync-ryzom-assets.mjs. */
type RyzomCatalogEntry = {
  id: string;
  code: string;
  label: string;
  kind: 'pc' | 'npc';
  clips: number;
  ambientClips: string[];
  meshes: number;
  triangles: number;
  joints: number;
};

const CATALOG_URL = '/shado/ryzom/ryzom-catalog.json';
const ASSET_ROOT = '/shado/ryzom/';

/**
 * Ryzom actors in the Shado VAT showcase.
 *
 * These are merged actors from the Ryzom intake (`npm run ryzom:actors`): one
 * GLB per actor carrying every outfit and face on a single unified skin, plus
 * every animation clip. They load through the showcase's canonical `sourceUrl`
 * path, so each is VAT-baked in the browser exactly like any other GLB.
 *
 * Run `node scripts/sync-ryzom-assets.mjs` to stage them; this pane reports what
 * is missing rather than failing blank if that has not been done.
 */
export class RyzomPlayground {
  public static async CreateScene(
    engine: BABYLON.Engine,
    canvas: HTMLCanvasElement
  ): Promise<BABYLON.Scene> {
    const scene = new BABYLON.Scene(engine);
    scene.clearColor = BABYLON.Color4.FromHexString('#101a14ff');
    (globalThis as any).__shadoScene = scene;
    delete (globalThis as any).__shadoWorld;

    const engineInstrumentation = new EngineInstrumentation(engine);
    engineInstrumentation.captureGPUFrameTime = true;

    // Ryzom actors stand about 2 m tall in the corrected Y-up orientation, so
    // this frames a small group rather than the EQ showcase's 54 m crowd shot.
    const camera = new BABYLON.ArcRotateCamera(
      'ryzom-camera',
      -Math.PI / 2,
      1.15,
      12,
      new BABYLON.Vector3(0, 1.0, 0),
      scene
    );
    camera.attachControl(canvas, true);
    camera.lowerRadiusLimit = 2;
    camera.upperRadiusLimit = 120;
    camera.wheelPrecision = 60;
    camera.panningSensibility = 80;

    const sky = new BABYLON.HemisphericLight('sky', new BABYLON.Vector3(0.25, 1, 0.1), scene);
    sky.intensity = 1.05;
    const sun = new BABYLON.DirectionalLight('sun', new BABYLON.Vector3(-0.45, -1, 0.35), scene);
    sun.intensity = 0.65;

    createShadoShowcaseEnvironment(BABYLON, scene);

    const catalog = await fetch(CATALOG_URL)
      .then((response) => (response.ok ? (response.json() as Promise<RyzomCatalogEntry[]>) : null))
      .catch(() => null);

    if (!catalog?.length) {
      showSetupHint(canvas);
      return scene;
    }

    // `?actors=zo-m,tr-chie` pins the set; the showcase otherwise auto-loads,
    // and every actor here costs a full in-browser VAT bake.
    const requested = new URLSearchParams(location.search).get('actors');
    const wanted = requested ? new Set(requested.split(',').map((s) => s.trim())) : null;
    const selected = wanted ? catalog.filter((entry) => wanted.has(entry.code)) : catalog;

    const models: ShadoVatShowcaseModel[] = selected.map((entry) => ({
      code: entry.code,
      label: entry.label,
      kind: entry.kind,
      // Only used to pick a nameplate colour; Ryzom has no EQ race mapping.
      nameRace: entry.kind === 'pc' ? 'human' : 'demon',
      scale: 1,
      // Ryzom clip names are French (marche, course, attente), so the
      // showcase's English ambient heuristic finds nothing. The sync script
      // picks these from the real clip list instead.
      ambientClips: entry.ambientClips,
      catalog: 'babylon',
      sourceUrl: `${ASSET_ROOT}${entry.id}.glb`,
    }));

    let ui: ReturnType<typeof createShadoVatShowcaseUi> | undefined;
    const controller = createShadoVatShowcase(scene, camera, {
      models,
      babylon: BABYLON,
      assetRoot: ASSET_ROOT,
      bakeWorkerUrl: '/shado/vat-bake-worker.js',
      // Ryzom actors carry far more clips than the EQ set (Fyros male has
      // 1,387), so each bake is heavier; two at a time keeps the tab responsive.
      bakeConcurrency: 2,
      autoLoad: !!requested,
      onStats: (stats: ShadoVatShowcaseStats) => ui?.update(stats),
    });

    (globalThis as any).__shadoShowcase = controller;
    (globalThis as any).__shadoShowcaseScene = scene;
    (globalThis as any).__ryzomCatalog = selected;

    ui = createShadoVatShowcaseUi(canvas, controller, {
      renderBackend: engine.isWebGPU ? 'WebGPU' : 'WebGL2',
      storageBackend: engine.isWebGPU ? 'StorageBuffer' : 'DataTexture',
      sample: () => {
        const gpuNanoseconds = engineInstrumentation.gpuFrameTimeCounter.current;
        return {
          fps: engine.getFps(),
          frameMs: engine.getDeltaTime(),
          gpuMs: gpuNanoseconds > 0 ? gpuNanoseconds / 1_000_000 : undefined,
        };
      },
    });

    scene.onDisposeObservable.add(() => {
      ui?.dispose();
      engineInstrumentation.dispose();
    });
    return scene;
  }
}

/** Staging is a separate manual step, so say so instead of rendering nothing. */
function showSetupHint(canvas: HTMLCanvasElement): void {
  const parent = canvas.parentElement ?? document.body;
  if (getComputedStyle(parent).position === 'static') parent.style.position = 'relative';
  const hint = document.createElement('div');
  hint.style.cssText = [
    'position:absolute',
    'inset:0',
    'display:flex',
    'flex-direction:column',
    'align-items:center',
    'justify-content:center',
    'gap:10px',
    'z-index:30',
    'color:#cfe3d4',
    'font:500 13px/1.6 system-ui',
    'text-align:center',
    'pointer-events:none',
  ].join(';');
  hint.innerHTML = `
    <div style="font:650 15px system-ui">No Ryzom actors staged</div>
    <div>Build them, then stage them into the sandbox:</div>
    <code style="background:#0c1410;border:1px solid #24382c;border-radius:6px;padding:8px 12px;color:#9fe0b4">
      npm run ryzom:actors<br>node scripts/sync-ryzom-assets.mjs
    </code>
    <div style="opacity:.7">Expected at <code>${CATALOG_URL}</code></div>
  `;
  parent.appendChild(hint);
}
