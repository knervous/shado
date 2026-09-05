import * as BABYLON from '@babylonjs/core';
import * as GUI from '@babylonjs/gui';
import { FontAsset } from '@babylonjs/addons/msdfText/fontAsset';
import { GridMaterial } from '@babylonjs/materials/grid';
import { EngineInstrumentation } from '@babylonjs/core/Instrumentation/engineInstrumentation';
import { SceneInstrumentation } from '@babylonjs/core/Instrumentation/sceneInstrumentation';
import {
  ShadoDynamicEntityContainer,
  ShadoDynamicEntityRenderer,
  ShadoSprite2DRenderer,
  ShadoText2DRenderer,
  type ShadoDynamicEntityInput,
  type ShadoSpriteAlphaMode,
  type ShadoSprite2DInput,
  type ShadoTextureAtlas,
} from '@knervous/shado/render';
import { Sprite2DMotionWorkerPool } from './Sprite2DMotionWorkerPool';

const ART_SIZE = 96;
const DEMO_SPRITE_COUNT = 13;
const ART_KEYS = [
  'cyan', 'coral', 'lime', 'violet', 'marker', 'mist',
  'orb', 'diamond', 'arrow', 'star', 'crate', 'ring',
] as const;
const LOAD_ART_KEYS = [
  'cyan', 'coral', 'lime', 'violet', 'marker',
  'orb', 'diamond', 'arrow', 'star', 'crate', 'ring',
] as const;

function makeDiagnosticAtlas(scene: BABYLON.Scene): ShadoTextureAtlas {
  const layerBytes = ART_SIZE * ART_SIZE * 4;
  const data = new Uint8Array(layerBytes * ART_KEYS.length);
  const colors = [
    [53, 211, 255], [255, 102, 91], [154, 231, 92], [181, 113, 255],
  ] as const;

  const pixel = (layer: number, x: number, y: number, rgba: readonly number[]) => {
    const offset = layer * layerBytes + (y * ART_SIZE + x) * 4;
    data[offset] = rgba[0];
    data[offset + 1] = rgba[1];
    data[offset + 2] = rgba[2];
    data[offset + 3] = rgba[3];
  };

  for (let layer = 0; layer < 4; layer++) {
    const color = colors[layer];
    for (let y = 0; y < ART_SIZE; y++) {
      for (let x = 0; x < ART_SIZE; x++) {
        const nx = (x + 0.5) / ART_SIZE * 2 - 1;
        const ny = 1 - (y + 0.5) / ART_SIZE;
        const head = Math.hypot(nx * 1.75, (ny - 0.76) * 3.4) < 0.52;
        const body = Math.abs(nx) < 0.36 + ny * 0.08 && ny > 0.19 && ny < 0.68;
        const legs = ny <= 0.25 && ny >= 0.04 &&
          (Math.abs(nx - 0.19) < 0.14 || Math.abs(nx + 0.19) < 0.14);
        const arm = nx > 0.28 && nx < 0.76 && ny > 0.42 && ny < 0.54;
        if (!(head || body || legs || arm)) continue;
        const border = Math.min(x, ART_SIZE - 1 - x, y, ART_SIZE - 1 - y);
        const alpha = border === 0 ? 150 : 255;
        const eye = head && nx > 0.12 && nx < 0.21 && ny > 0.75 && ny < 0.81;
        pixel(layer, x, y, eye ? [10, 20, 28, alpha] : [...color, alpha]);
      }
    }
  }

  for (let y = 0; y < ART_SIZE; y++) {
    for (let x = 0; x < ART_SIZE; x++) {
      const nx = (x + 0.5) / ART_SIZE * 2 - 1;
      const ny = (y + 0.5) / ART_SIZE * 2 - 1;
      const radius = Math.hypot(nx, ny);
      if (radius > 0.55 && radius < 0.84) pixel(4, x, y, [255, 207, 64, 245]);
      if (Math.abs(nx) < 0.045 || Math.abs(ny) < 0.045) {
        pixel(4, x, y, [255, 255, 255, 230]);
      }
      const mistAlpha = Math.max(0, Math.min(150, Math.round((1 - radius) * 180)));
      if (mistAlpha) pixel(5, x, y, [92, 170, 255, mistAlpha]);
    }
  }

  const extraColors = [
    [58, 221, 196], [255, 125, 198], [255, 164, 62],
    [255, 224, 91], [105, 168, 255], [202, 122, 255],
  ] as const;
  for (let y = 0; y < ART_SIZE; y++) {
    for (let x = 0; x < ART_SIZE; x++) {
      const nx = (x + 0.5) / ART_SIZE * 2 - 1;
      const ny = (y + 0.5) / ART_SIZE * 2 - 1;
      const radius = Math.hypot(nx, ny);
      const angle = Math.atan2(ny, nx);
      const masks = [
        radius < 0.72,
        Math.abs(nx) + Math.abs(ny) < 0.93,
        (nx > -0.68 && nx < 0.18 && Math.abs(ny) < 0.2) ||
          (nx >= -0.02 && nx < 0.76 && Math.abs(ny) < 0.78 - nx),
        radius < (Math.cos(angle * 5) > 0 ? 0.82 : 0.38),
        Math.max(Math.abs(nx), Math.abs(ny)) < 0.72,
        radius > 0.48 && radius < 0.78,
      ];
      for (let extra = 0; extra < masks.length; extra++) {
        if (!masks[extra]) continue;
        const color = extraColors[extra];
        const highlight = extra === 0 && nx < -0.18 && ny < -0.18 && radius < 0.58;
        const crateLine = extra === 4 &&
          (Math.abs(Math.abs(nx) - Math.abs(ny)) < 0.065 || Math.max(Math.abs(nx), Math.abs(ny)) > 0.62);
        pixel(6 + extra, x, y, highlight
          ? [205, 255, 248, 255]
          : crateLine ? [35, 72, 116, 255] : [...color, 255]);
      }
    }
  }

  const texture = new BABYLON.RawTexture2DArray(
    data,
    ART_SIZE,
    ART_SIZE,
    ART_KEYS.length,
    BABYLON.Engine.TEXTUREFORMAT_RGBA,
    scene,
    false,
    false,
    BABYLON.Texture.NEAREST_SAMPLINGMODE,
    BABYLON.Engine.TEXTURETYPE_UNSIGNED_BYTE
  );
  texture.name = 'sprites-2d-diagnostic-atlas';
  texture.wrapU = BABYLON.Texture.CLAMP_ADDRESSMODE;
  texture.wrapV = BABYLON.Texture.CLAMP_ADDRESSMODE;
  const entries = Object.fromEntries(ART_KEYS.map((key, layer) => [
    key,
    { layer, rect: { u0: 0, v0: 0, u1: 1, v1: 1 } },
  ]));
  return {
    texture: texture as any,
    entries,
    get: key => entries[key] ?? entries.cyan,
    dispose: () => texture.dispose(),
  };
}

type BatchOptions = {
  alphaMode?: ShadoSpriteAlphaMode;
};

type Sprite2DLoadScenario = 'field' | 'dense';

type Sprite2DLoadTestResult = {
  path: 'optimized' | 'full';
  backend: 'webgl2' | 'webgpu';
  scenario: Sprite2DLoadScenario;
  sprites: number;
  submittedSprites: number;
  batches: number;
  drawCalls: number;
  recordBytes: number;
  workingSetBytes: number;
  gpuCapacityBytes: number;
  setupMs: number;
  firstFrameMs: number;
  measuredFrames: number;
  frameMs: { mean: number; p50: number; p95: number; max: number };
  gpuMs: { mean: number; p50: number; p95: number; max: number } | null;
  intervalMs: { mean: number; p50: number; p95: number; max: number };
  effectiveFps: number;
  gpuCulling: boolean;
  indirectDraw: boolean;
};

function loadPopulation(count: number, scenario: Sprite2DLoadScenario): {
  cutout: ShadoSprite2DInput[];
  mist: ShadoSprite2DInput[];
} {
  const cutout: ShadoSprite2DInput[] = [];
  const mist: ShadoSprite2DInput[] = [];
  const columns = Math.ceil(Math.sqrt(count));
  const rows = Math.ceil(count / columns);
  const spacing = scenario === 'field' ? 1.2 : Math.min(20 / columns, 14 / rows);
  const size = scenario === 'field' ? 0.72 : Math.max(0.006, spacing * 0.72);

  for (let index = 0; index < count; index++) {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const artKey = index % 13 === 0
      ? 'mist'
      : LOAD_ART_KEYS[(index * 7) % LOAD_ART_KEYS.length];
    const widthVariation = 0.72 + ((index * 37) % 101) / 100 * 0.58;
    const heightVariation = 0.72 + ((index * 61 + 17) % 101) / 100 * 0.58;
    const input: ShadoSprite2DInput = {
      id: `load-${index}`,
      textureKey: artKey,
      position: [
        (column - (columns - 1) * 0.5) * spacing,
        (row - (rows - 1) * 0.5) * spacing,
      ],
      size: [size * widthVariation, size * heightVariation],
      rotationRad: ((index * 43) % 360) * Math.PI / 180,
      opacity: artKey === 'mist' ? 0.62 : 1,
      layer: artKey === 'mist' ? 3 : 2,
      order: index,
      minPixelSize: 0,
    };
    (artKey === 'mist' ? mist : cutout).push(input);
  }
  return { cutout, mist };
}

function summarize(values: readonly number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const percentile = (fraction: number) =>
    sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))] ?? 0;
  return {
    mean: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0,
    p50: percentile(0.5),
    p95: percentile(0.95),
    max: sorted.at(-1) ?? 0,
  };
}

type Sprite2DControlState = {
  path: 'optimized' | 'full';
  population: number;
  layout: Sprite2DLoadScenario;
  tileSize: number;
  lod: number;
  animation: boolean;
  motionSeed: number;
  motionSpeed: number;
  motionCadence: number;
  motionWorkers: number;
  gpuMotion: boolean;
  gpuCulling: boolean;
  grid: boolean;
  text: string;
  fontSize: number;
  align: 'left' | 'center' | 'right';
};

function createSprite2DControlOverlay(
  initial: Sprite2DControlState,
  handlers: {
    spawn(count: number, layout: Sprite2DLoadScenario): void;
    setLod(value: number): void;
    setAnimation(value: boolean): void;
    setMotion(seed: number, speed: number, cadence: number): void;
    setMotionWorkers(value: number): void;
    setGpuCulling(value: boolean): void;
    setGrid(value: boolean): void;
    setText(text: string, fontSize: number, align: 'left' | 'center' | 'right'): void;
    resetView(): void;
    setPath(path: 'optimized' | 'full'): void;
    setTileSize(value: number): void;
  }
) {
  const panel = document.createElement('section');
  panel.className = 'sprites-2d-controls';
  panel.innerHTML = `
    <style>
      .sprites-2d-controls{position:fixed;z-index:20;right:16px;top:58px;width:300px;max-height:calc(100vh - 126px);overflow-x:hidden;overflow-y:auto;padding:14px;border:1px solid #345064;border-radius:9px;background:#071018e8;color:#dcebf5;box-shadow:0 12px 32px #0008;font:12px system-ui,sans-serif;text-align:left;box-sizing:border-box}
      .sprites-2d-controls *{box-sizing:border-box;min-width:0}.sprites-2d-controls .head{display:flex;align-items:center;justify-content:space-between;gap:8px}.sprites-2d-controls h2{margin:0 0 3px;font-size:17px}.sprites-2d-controls p{margin:0 0 11px;color:#86a4b8;font-size:11px}.sprites-2d-controls fieldset{min-inline-size:0;margin:0 0 10px;padding:9px;border:1px solid #263d4d;border-radius:7px}.sprites-2d-controls legend{padding:0 5px;color:#8fd6ff;font-size:10px;text-transform:uppercase;letter-spacing:.08em}.sprites-2d-controls label{display:grid;grid-template-columns:88px minmax(0,1fr);align-items:center;gap:7px;margin:6px 0;color:#aec5d4}.sprites-2d-controls label span{display:grid;grid-template-columns:minmax(0,1fr) 34px;align-items:center;gap:5px}.sprites-2d-controls input,.sprites-2d-controls select,.sprites-2d-controls button{width:100%;height:29px;border:1px solid #385468;border-radius:5px;background:#102331;color:#eef8ff;font:11px inherit}.sprites-2d-controls input[type=text]{padding:0 7px}.sprites-2d-controls input[type=range]{border:0}.sprites-2d-controls input[type=checkbox]{width:17px;height:17px;justify-self:start}.sprites-2d-controls button{padding:0 8px;cursor:pointer}.sprites-2d-controls button:hover{border-color:#76cfff;background:#173448}.sprites-2d-controls .collapse{width:31px;font-size:16px}.sprites-2d-controls .spawn-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.sprites-2d-controls output{color:#fff;font:11px ui-monospace,monospace}.sprites-2d-controls .live{white-space:pre-line;color:#91e6ad;line-height:1.45}.sprites-2d-controls .row-actions{display:grid;grid-template-columns:1fr 1fr;gap:6px}.sprites-2d-controls.is-collapsed{width:190px}.sprites-2d-controls.is-collapsed>p,.sprites-2d-controls.is-collapsed>fieldset,.sprites-2d-controls.is-collapsed>.row-actions,.sprites-2d-controls.is-collapsed>.live{display:none}@media(max-width:760px){.sprites-2d-controls{top:auto;right:10px;bottom:58px;width:270px;max-height:46vh}}
    </style>
    <div class="head"><h2>2D renderer controls</h2><button class="collapse" data-role="collapse" title="Collapse controls">−</button></div><p>Locked world-Y canvas · pan and zoom only</p>
    <fieldset><legend>Renderer</legend>
      <label>Path <select data-role="path"><option value="optimized">Optimized 2D</option><option value="full">Full backup</option></select></label>
      <label>Tile size <select data-role="tile"><option>4</option><option>8</option><option>16</option><option>32</option></select></label>
      <label>LOD pixels <span><input data-role="lod" type="range" min="0" max="5" step="0.25"><output data-role="lod-value"></output></span></label>
    </fieldset>
    <fieldset><legend>Population</legend>
      <label>Layout <select data-role="layout"><option value="field">World field</option><option value="dense">Dense viewport</option></select></label>
      <label>Resident <output data-role="population-value"></output></label>
      <div class="spawn-grid"><button data-role="demo">Demo</button><button data-add="100">+100</button><button data-add="1000">+1K</button><button data-add="10000">+10K</button><button data-add="100000">+100K</button><button data-add="250000">+250K</button></div>
      <label>Seeded motion <input data-role="animation" type="checkbox"></label>
      <label>Motion seed <input data-role="motion-seed" type="number" step="1"></label>
      <label>Speed <span><input data-role="motion-speed" type="range" min="0" max="4" step="0.05"><output data-role="speed-value"></output></span></label>
      <label>Vector change <select data-role="motion-cadence"><option value="0.5">0.5 sec</option><option value="1">1 sec</option><option value="2">2 sec</option><option value="4">4 sec</option></select></label>
      <label>Worker shards <select data-role="motion-workers"><option>1</option><option>2</option><option>4</option><option>8</option></select></label>
      <label>GPU culling <input data-role="gpu-culling" type="checkbox"></label>
      <label>Grid <input data-role="grid" type="checkbox"></label>
    </fieldset>
    <fieldset><legend>MSDF text</legend>
      <label>Content <input data-role="text" type="text"></label>
      <label>Font size <span><input data-role="font-size" type="range" min="0.35" max="2.5" step="0.05"><output data-role="font-value"></output></span></label>
      <label>Alignment <select data-role="align"><option value="left">Left</option><option value="center">Center</option><option value="right">Right</option></select></label>
    </fieldset>
    <div class="row-actions"><button data-role="reset">Reset view</button><button data-role="rerun">Rebuild population</button></div>
    <p class="live" data-role="stats">Preparing renderer…</p>`;
  document.body.appendChild(panel);

  const element = <T extends HTMLElement>(role: string) =>
    panel.querySelector<T>(`[data-role="${role}"]`)!;
  const path = element<HTMLSelectElement>('path');
  const tile = element<HTMLSelectElement>('tile');
  const lod = element<HTMLInputElement>('lod');
  const lodValue = element<HTMLOutputElement>('lod-value');
  const layout = element<HTMLSelectElement>('layout');
  const populationValue = element<HTMLOutputElement>('population-value');
  const animation = element<HTMLInputElement>('animation');
  const motionSeed = element<HTMLInputElement>('motion-seed');
  const motionSpeed = element<HTMLInputElement>('motion-speed');
  const speedValue = element<HTMLOutputElement>('speed-value');
  const motionCadence = element<HTMLSelectElement>('motion-cadence');
  const motionWorkers = element<HTMLSelectElement>('motion-workers');
  const gpuCulling = element<HTMLInputElement>('gpu-culling');
  const grid = element<HTMLInputElement>('grid');
  const text = element<HTMLInputElement>('text');
  const fontSize = element<HTMLInputElement>('font-size');
  const fontValue = element<HTMLOutputElement>('font-value');
  const align = element<HTMLSelectElement>('align');
  path.value = initial.path;
  tile.value = String(initial.tileSize);
  lod.value = String(initial.lod);
  lodValue.value = initial.lod.toFixed(2);
  layout.value = initial.layout;
  populationValue.value = initial.population.toLocaleString();
  animation.checked = initial.animation;
  motionSeed.value = String(initial.motionSeed);
  motionSpeed.value = String(initial.motionSpeed);
  speedValue.value = initial.motionSpeed.toFixed(2);
  motionCadence.value = String(initial.motionCadence);
  motionWorkers.value = String(initial.motionWorkers);
  motionWorkers.disabled = initial.gpuMotion;
  motionWorkers.title = initial.gpuMotion
    ? 'WebGPU motion runs in compute; worker shards are the fallback setting.'
    : 'Copy-isolated SIMD worker shards.';
  gpuCulling.checked = initial.gpuCulling;
  gpuCulling.disabled = !initial.gpuMotion;
  gpuCulling.title = initial.gpuMotion
    ? 'Compact visible cutout sprites on the GPU and use indirect drawing when Babylon exposes it.'
    : 'GPU culling requires WebGPU motion ownership.';
  grid.checked = initial.grid;
  text.value = initial.text;
  fontSize.value = String(initial.fontSize);
  fontValue.value = initial.fontSize.toFixed(2);
  align.value = initial.align;
  const collapse = element<HTMLButtonElement>('collapse');
  collapse.addEventListener('click', () => {
    const collapsed = panel.classList.toggle('is-collapsed');
    collapse.textContent = collapsed ? '+' : '−';
    collapse.title = collapsed ? 'Expand controls' : 'Collapse controls';
  });

  let population = initial.population;
  const spawn = () => {
    populationValue.value = population.toLocaleString();
    handlers.spawn(population, layout.value as Sprite2DLoadScenario);
  };
  element<HTMLButtonElement>('demo').addEventListener('click', () => {
    population = DEMO_SPRITE_COUNT;
    spawn();
  });
  for (const button of Array.from(panel.querySelectorAll<HTMLButtonElement>('[data-add]'))) {
    button.addEventListener('click', () => {
      population += Number(button.dataset.add);
      spawn();
    });
  }
  element<HTMLButtonElement>('rerun').addEventListener('click', spawn);
  element<HTMLButtonElement>('reset').addEventListener('click', handlers.resetView);
  layout.addEventListener('change', spawn);
  path.addEventListener('change', () => handlers.setPath(path.value as 'optimized' | 'full'));
  tile.addEventListener('change', () => handlers.setTileSize(Number(tile.value)));
  lod.addEventListener('input', () => {
    lodValue.value = Number(lod.value).toFixed(2);
    handlers.setLod(Number(lod.value));
  });
  animation.addEventListener('change', () => handlers.setAnimation(animation.checked));
  const updateMotion = () => handlers.setMotion(
    Math.trunc(Number(motionSeed.value) || 0),
    Number(motionSpeed.value),
    Number(motionCadence.value)
  );
  motionSeed.addEventListener('change', updateMotion);
  motionSpeed.addEventListener('input', () => {
    speedValue.value = Number(motionSpeed.value).toFixed(2);
    updateMotion();
  });
  motionCadence.addEventListener('change', updateMotion);
  motionWorkers.addEventListener('change', () => handlers.setMotionWorkers(Number(motionWorkers.value)));
  gpuCulling.addEventListener('change', () => handlers.setGpuCulling(gpuCulling.checked));
  grid.addEventListener('change', () => handlers.setGrid(grid.checked));
  const updateText = () => handlers.setText(
    text.value,
    Number(fontSize.value),
    align.value as 'left' | 'center' | 'right'
  );
  text.addEventListener('input', updateText);
  fontSize.addEventListener('input', () => {
    fontValue.value = Number(fontSize.value).toFixed(2);
    updateText();
  });
  align.addEventListener('change', updateText);

  return {
    updateStats(value: string) { element<HTMLElement>('stats').textContent = value; },
    dispose() { panel.remove(); },
  };
}

export class Sprite2DPlayground {
  public static async CreateScene(
    engine: BABYLON.Engine,
    canvas: HTMLCanvasElement
  ): Promise<BABYLON.Scene> {
    const scene = new BABYLON.Scene(engine);
    const params = new URLSearchParams(window.location.search);
    const loadTestEnabled = params.get('loadTest') === '1';
    const loadScenario: Sprite2DLoadScenario = params.get('scenario') === 'dense'
      ? 'dense'
      : 'field';
    const loadCount = Math.max(1, Math.min(1_000_000, Number(params.get('count')) || 10_000));
    const loadWarmupFrames = Math.max(1, Number(params.get('warmupFrames')) || 20);
    const loadSampleFrames = Math.max(10, Number(params.get('sampleFrames')) || 90);
    const interactiveCount = Math.max(
      DEMO_SPRITE_COUNT,
      Math.min(1_000_000, Number(params.get('spawn')) || DEMO_SPRITE_COUNT)
    );
    const interactiveLayout: Sprite2DLoadScenario = params.get('layout') === 'dense'
      ? 'dense'
      : 'field';
    const configuredTileSize = [4, 8, 16, 32].includes(Number(params.get('tileSize')))
      ? Number(params.get('tileSize'))
      : 8;
    const requestedLod = Number(params.get('lod'));
    const configuredLod = params.has('lod') && Number.isFinite(requestedLod)
      ? Math.max(0, Math.min(5, requestedLod))
      : 0.75;
    const configuredMotionSeed = Math.trunc(Number(params.get('motionSeed')) || 1337);
    const loadGpuMotion = loadTestEnabled && params.get('gpuMotion') === '1';
    const gpuCullingEnabled = params.get('gpuCulling') === '1';
    const requestedSpeed = Number(params.get('motionSpeed'));
    const configuredMotionSpeed = params.has('motionSpeed') && Number.isFinite(requestedSpeed)
      ? Math.max(0, Math.min(4, requestedSpeed))
      : 0.8;
    const requestedCadence = Number(params.get('motionCadence'));
    const configuredMotionCadence = [0.5, 1, 2, 4].includes(requestedCadence)
      ? requestedCadence
      : 2;
    const availableMotionWorkers = Math.max(1, navigator.hardwareConcurrency - 1 || 1);
    const defaultMotionWorkers = availableMotionWorkers >= 4 ? 4 : availableMotionWorkers >= 2 ? 2 : 1;
    const requestedMotionWorkers = Number(params.get('motionWorkers'));
    const configuredMotionWorkers = [1, 2, 4, 8].includes(requestedMotionWorkers)
      ? requestedMotionWorkers
      : defaultMotionWorkers;
    const loadInstrumentation = loadTestEnabled ? new EngineInstrumentation(engine) : undefined;
    const loadSceneInstrumentation = loadTestEnabled ? new SceneInstrumentation(scene) : undefined;
    if (loadInstrumentation) loadInstrumentation.captureGPUFrameTime = true;
    scene.clearColor = BABYLON.Color4.FromHexString('#071018ff');
    const camera = new BABYLON.FreeCamera(
      'sprites-2d-camera',
      new BABYLON.Vector3(0, 20, 0),
      scene
    );
    camera.mode = BABYLON.Camera.ORTHOGRAPHIC_CAMERA;
    camera.upVector.set(0, 0, 1);
    camera.setTarget(BABYLON.Vector3.Zero());
    // Custom planar controls preserve the exact top-down world orientation:
    // drag changes only X/Z and the wheel changes only orthographic scale.
    let centerX = 0;
    let centerZ = 0;
    let zoomScale = 1;
    let activePointer: number | undefined;
    let pointerX = 0;
    let pointerY = 0;
    let pointerTravel = 0;
    let pickReadout: GUI.TextBlock | undefined;
    let previousAspect = 0;
    let previousZoomScale = 0;
    const updateLockedProjection = () => {
      const aspect = Math.max(0.1, engine.getRenderWidth() / engine.getRenderHeight());
      if (
        Math.abs(aspect - previousAspect) < 0.0001 &&
        Math.abs(zoomScale - previousZoomScale) < 0.0001
      ) return;
      previousAspect = aspect;
      previousZoomScale = zoomScale;
      const halfHeight = Math.max(9, 12 / aspect) * zoomScale;
      camera.orthoTop = halfHeight;
      camera.orthoBottom = -halfHeight;
      camera.orthoLeft = -halfHeight * aspect;
      camera.orthoRight = halfHeight * aspect;
    };
    const updateCameraCenter = () => {
      camera.position.set(centerX, 20, centerZ);
      camera.setTarget(new BABYLON.Vector3(centerX, 0, centerZ));
    };
    const onPointerDown = (event: PointerEvent) => {
      if (activePointer !== undefined || event.button !== 0) return;
      activePointer = event.pointerId;
      pointerX = event.clientX;
      pointerY = event.clientY;
      pointerTravel = 0;
      canvas.setPointerCapture(event.pointerId);
    };
    const onPointerMove = (event: PointerEvent) => {
      if (event.pointerId !== activePointer) return;
      const width = Math.max(1, engine.getRenderWidth());
      const height = Math.max(1, engine.getRenderHeight());
      const worldWidth = Math.abs((camera.orthoRight ?? 1) - (camera.orthoLeft ?? -1));
      const worldHeight = Math.abs((camera.orthoTop ?? 1) - (camera.orthoBottom ?? -1));
      pointerTravel += Math.hypot(event.clientX - pointerX, event.clientY - pointerY);
      centerX -= (event.clientX - pointerX) * worldWidth / width;
      centerZ += (event.clientY - pointerY) * worldHeight / height;
      pointerX = event.clientX;
      pointerY = event.clientY;
      updateCameraCenter();
    };
    const onPointerUp = async (event: PointerEvent) => {
      if (event.pointerId !== activePointer) return;
      activePointer = undefined;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      if (pointerTravel <= 4 && (optimizedRenderers.length || textRenderers.length) && pickReadout) {
        const rect = canvas.getBoundingClientRect();
        const pointerArgs = [
          event.clientX - rect.left,
          event.clientY - rect.top,
          rect.width,
          rect.height,
        ] as const;
        const textHit = textRenderers[0]?.pickScreen(...pointerArgs);
        const spriteHit = optimizedRenderers[0]?.pickScreen(...pointerArgs);
        const gpuRenderer = optimizedRenderers.find(renderer => renderer.isGpuMotionEnabled);
        const selectedSnapshot = !textHit && gpuRenderer
          ? await gpuRenderer.readCpuPositions({ tier: 'selected' })
          : undefined;
        const selectedGpu = selectedSnapshot?.entries[0];
        pickReadout.text = textHit
          ? `Text: ${textHit.id} · “${textHit.text.text.replace(/\n/g, ' ')}”`
          : spriteHit
            ? `Pick: ${spriteHit.id} · UV ${spriteHit.uv[0].toFixed(2)}, ${spriteHit.uv[1].toFixed(2)}`
          : selectedGpu
            ? `${selectedSnapshot?.stale ? 'Stale' : 'GPU'} selected: ${selectedGpu.id} · ${selectedGpu.position[0].toFixed(2)}, ${selectedGpu.position[1].toFixed(2)}${selectedSnapshot?.inBand ? '' : ' · async'}`
          : gpuRenderer
            ? 'GPU motion: no selected entity to read lazily'
            : 'Pick: empty canvas';
      }
    };
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const next = zoomScale * Math.exp(Math.max(-4, Math.min(4, event.deltaY * 0.001)));
      // There is deliberately no zoom-out ceiling. Clamp only pathological
      // floating-point overflow while preserving a practical infinite canvas.
      zoomScale = Math.max(0.02, Number.isFinite(next) ? next : Number.MAX_VALUE / 1024);
      updateLockedProjection();
    };
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointercancel', onPointerUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    updateLockedProjection();
    scene.onBeforeRenderObservable.add(updateLockedProjection);
    const ground = BABYLON.MeshBuilder.CreateGround(
      'sprites-2d-grid',
      { width: 100, height: 100, subdivisions: 2 },
      scene
    );
    const grid = new GridMaterial('sprites-2d-grid-material', scene);
    grid.majorUnitFrequency = 5;
    grid.minorUnitVisibility = 0.42;
    grid.gridRatio = 1;
    grid.mainColor = BABYLON.Color3.FromHexString('#0d1822');
    grid.lineColor = BABYLON.Color3.FromHexString('#35536a');
    grid.opacity = 0.96;
    ground.material = grid;
    ground.setEnabled(params.get('grid') !== '0');
    scene.onBeforeRenderObservable.add(() => {
      const scale = Math.max(1, Math.min(Number.MAX_VALUE / 1024, zoomScale * 1.5));
      ground.scaling.set(scale, 1, scale);
    });

    const atlas = makeDiagnosticAtlas(scene);
    let fontAsset: FontAsset | undefined;
    if (!loadTestEnabled) {
      try {
        const definition = await fetch('https://assets.babylonjs.com/fonts/roboto-regular.json')
          .then(response => response.text());
        fontAsset = new FontAsset(
          definition,
          'https://assets.babylonjs.com/fonts/roboto-regular.png',
          scene as any
        );
      } catch (error) {
        console.warn('[sprites-2d] MSDF font could not be loaded', error);
      }
    }
    const useLegacyPath = params.get('path') === 'legacy';
    const batches: Array<{ container: ShadoDynamicEntityContainer; renderer: ShadoDynamicEntityRenderer }> = [];
    const makeBatch = (name: string, options: BatchOptions, entities: ShadoDynamicEntityInput[]) => {
      const container = new ShadoDynamicEntityContainer(engine, atlas);
      const renderer = new ShadoDynamicEntityRenderer(scene as any, container, atlas, {
        presentation: 'ground',
        pivot: [0.5, 0.5],
        alphaMode: options.alphaMode,
        alphaCutoff: options.alphaMode === 'premultiplied' ? 0.01 : 0.35,
      });
      renderer.mesh.name = name;
      container.upsertMany(entities);
      batches.push({ container, renderer });
      return container;
    };
    const toLegacy = (sprite: ShadoSprite2DInput): ShadoDynamicEntityInput => ({
      id: sprite.id,
      textureKey: sprite.textureKey,
      x: sprite.position[0],
      y: sprite.position[1],
      z: (sprite.layer ?? 0) * 0.01,
      width: sprite.size[0],
      height: sprite.size[1],
      depth: 0.01,
      rotationRad: sprite.rotationRad,
      rotationDeg: sprite.rotationDeg,
      opacity: sprite.opacity,
      selected: sprite.selected,
      highlighted: sprite.highlighted,
      sortKey: sprite.order,
    });
    const demoCutoutSprites: ShadoSprite2DInput[] = [
      { id: 'short-wide', textureKey: 'cyan', position: [-5.2, 2.6], size: [2.25, 2.6], selected: true, rotationDeg: -8, layer: 2 },
      { id: 'tall-thin', textureKey: 'coral', position: [-1.9, 2.1], size: [1.25, 4.3], highlighted: true, rotationDeg: 12, layer: 2 },
      { id: 'moving', textureKey: 'lime', position: [2.3, 2.2], size: [1.65, 3.35], rotationDeg: -18, layer: 2, order: 2 },
      { id: 'violet', textureKey: 'violet', position: [5.8, 2.7], size: [1.6, 2.7], rotationDeg: 24, layer: 2, order: 3 },
      { id: 'ground-marker', textureKey: 'marker', position: [-4.4, -3.4], size: [2.8, 2.8], rotationDeg: 20, layer: 1 },
      { id: 'orb', textureKey: 'orb', position: [-6.9, -0.6], size: [1.5, 1.5], layer: 2 },
      { id: 'diamond', textureKey: 'diamond', position: [-3.2, -0.9], size: [1.7, 1.7], rotationDeg: 14, layer: 2 },
      { id: 'arrow', textureKey: 'arrow', position: [-0.6, 0], size: [2.2, 1.5], rotationDeg: -18, layer: 2 },
      { id: 'star', textureKey: 'star', position: [2.2, -0.2], size: [1.8, 1.8], rotationDeg: 9, layer: 2 },
      { id: 'crate', textureKey: 'crate', position: [4.6, -0.5], size: [1.7, 1.7], rotationDeg: -8, layer: 2 },
      { id: 'ring', textureKey: 'ring', position: [7, -0.8], size: [1.8, 1.8], layer: 2 },
    ];
    const demoMistSprites: ShadoSprite2DInput[] = [
      { id: 'mist-far', textureKey: 'mist', position: [-0.2, -2.2], size: [5.8, 4.2], opacity: 0.72, layer: 3, order: 1 },
      { id: 'mist-near', textureKey: 'mist', position: [2.2, -3.5], size: [4.8, 3.4], opacity: 0.58, layer: 3, order: 2 },
    ];

    const population = loadTestEnabled
      ? loadPopulation(loadCount, loadScenario)
      : interactiveCount > DEMO_SPRITE_COUNT
        ? loadPopulation(interactiveCount, interactiveLayout)
        : { cutout: demoCutoutSprites, mist: demoMistSprites };
    const cutoutSprites = population.cutout;
    const mistSprites = population.mist;
    const setupStarted = performance.now();
    const optimizedRenderers: ShadoSprite2DRenderer[] = [];
    const textRenderers: ShadoText2DRenderer[] = [];
    if (useLegacyPath) {
      await ShadoDynamicEntityContainer.initialize(engine, { wasm: false });
      makeBatch('legacy-2d-cutout-layer', {}, cutoutSprites.map(toLegacy));
      makeBatch('legacy-2d-premultiplied-layer', { alphaMode: 'premultiplied' }, mistSprites.map(toLegacy));
    } else {
      const cutout = new ShadoSprite2DRenderer(scene as any, atlas, {
        alphaMode: 'cutout',
        tileSize: configuredTileSize,
        minPixelSize: loadTestEnabled ? 0 : configuredLod,
        gpuCulling: gpuCullingEnabled,
      });
      cutout.upsertMany(cutoutSprites);
      const mist = new ShadoSprite2DRenderer(scene as any, atlas, {
        alphaMode: 'premultiplied',
        alphaCutoff: 0.01,
        tileSize: configuredTileSize,
        minPixelSize: loadTestEnabled ? 0 : configuredLod,
      });
      mist.upsertMany(mistSprites);
      optimizedRenderers.push(cutout, mist);
    }
    if (fontAsset) {
      const text = new ShadoText2DRenderer(scene as any, fontAsset as any, {
        tileSize: configuredTileSize,
        minPixelSize: configuredLod,
      });
      text.upsertMany([
        {
          id: 'text-title',
          text: 'ARBITRARY MSDF TEXT',
          position: [4, 7.1],
          fontSize: 1.15,
          color: [0.66, 0.91, 1, 1],
          layer: 8,
        },
        {
          id: 'text-wrapped',
          text: 'Locked to the 2D surface\nwith wrapping, alignment,\nrotation, color, and zoom.',
          position: [0, -6.4],
          fontSize: 0.72,
          maxWidth: 9.5,
          lineHeight: 0.86,
          align: 'center',
          color: [1, 0.84, 0.47, 0.96],
          rotationDeg: -3,
          layer: 8,
        },
        {
          id: 'text-live',
          text: 'Runtime text 000',
          position: [6.2, -0.6],
          fontSize: 0.62,
          pivot: [1, 0.5],
          color: [0.67, 1, 0.66, 1],
          layer: 9,
        },
      ]);
      textRenderers.push(text);
    }
    // Keep the existing static load benchmark comparable. Interactive stress
    // populations exercise compute motion; benchmark mode measures rendering.
    const useGpuMotion = engine.isWebGPU && !useLegacyPath && (!loadTestEnabled || loadGpuMotion);
    const motionPool = loadTestEnabled || useGpuMotion
      ? undefined
      : new Sprite2DMotionWorkerPool(configuredMotionWorkers);
    const setupMs = performance.now() - setupStarted;

    let activePopulationCount = cutoutSprites.length + mistSprites.length;
    let activeLayout = loadTestEnabled ? loadScenario : interactiveLayout;
    let animationEnabled = loadGpuMotion || activePopulationCount <= DEMO_SPRITE_COUNT;
    let motionSeed = configuredMotionSeed;
    let motionSpeed = configuredMotionSpeed;
    let motionCadence = configuredMotionCadence;
    let motionEntries: Array<{ id: string; batch: 0 | 1 }> = [];
    let motionBounds: [number, number, number, number] = [-10, -7, 10, 7];
    let previousMotionTime = performance.now();
    const resetMotion = (next: { cutout: ShadoSprite2DInput[]; mist: ShadoSprite2DInput[] }) => {
      const now = performance.now();
      const all = [
        ...next.cutout.map(input => ({ input, batch: 0 as const })),
        ...next.mist.map(input => ({ input, batch: 1 as const })),
      ];
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      const positions = new Float32Array(all.length * 2);
      motionEntries = all.map(({ input, batch }, index) => {
        minX = Math.min(minX, input.position[0]);
        minY = Math.min(minY, input.position[1]);
        maxX = Math.max(maxX, input.position[0]);
        maxY = Math.max(maxY, input.position[1]);
        positions[index * 2] = input.position[0];
        positions[index * 2 + 1] = input.position[1];
        return { id: input.id, batch };
      });
      const padding = activeLayout === 'field' ? 3 : 0.75;
      motionBounds = Number.isFinite(minX)
        ? [minX - padding, minY - padding, maxX + padding, maxY + padding]
        : [-10, -7, 10, 7];
      const motionConfig = {
        seed: motionSeed,
        speed: motionSpeed,
        cadenceMs: motionCadence * 1000,
        bounds: motionBounds,
      } as const;
      if (useGpuMotion) {
        optimizedRenderers[0]?.enableGpuMotion(motionConfig, 0, now);
        optimizedRenderers[1]?.enableGpuMotion(motionConfig, next.cutout.length, now);
      } else {
        void motionPool?.setPopulation(positions, motionConfig, now);
      }
      previousMotionTime = now;
    };
    resetMotion(population);
    const replacePopulation = (count: number, layout: Sprite2DLoadScenario) => {
      const next = count <= DEMO_SPRITE_COUNT
        ? { cutout: demoCutoutSprites, mist: demoMistSprites }
        : loadPopulation(count, layout);
      if (useLegacyPath) {
        batches[0]?.container.clearEntities();
        batches[0]?.container.upsertMany(next.cutout.map(toLegacy));
        batches[1]?.container.clearEntities();
        batches[1]?.container.upsertMany(next.mist.map(toLegacy));
      } else {
        optimizedRenderers[0]?.clear();
        optimizedRenderers[0]?.upsertMany(next.cutout);
        optimizedRenderers[1]?.clear();
        optimizedRenderers[1]?.upsertMany(next.mist);
      }
      activePopulationCount = next.cutout.length + next.mist.length;
      activeLayout = layout;
      resetMotion(next);
      const url = new URL(window.location.href);
      if (count <= DEMO_SPRITE_COUNT) url.searchParams.delete('spawn');
      else url.searchParams.set('spawn', String(count));
      url.searchParams.set('layout', layout);
      window.history.replaceState(null, '', url);
    };

    let previousTextTick = -1;
    let motionStepInFlight = false;
    const runMotionStep = async (now: number, dt: number) => {
      if (!motionPool || motionStepInFlight) return;
      motionStepInFlight = true;
      try {
        const shards = await motionPool.step(now, dt);
        const updates: [Array<{ id: string; position: [number, number] }>, Array<{ id: string; position: [number, number] }>] = [[], []];
        for (const shard of shards) {
          for (let localIndex = 0; localIndex < shard.positions.length / 2; localIndex++) {
            const entry = motionEntries[shard.start + localIndex];
            if (!entry) continue;
            updates[entry.batch].push({
              id: entry.id,
              position: [shard.positions[localIndex * 2], shard.positions[localIndex * 2 + 1]],
            });
          }
        }
        if (useLegacyPath) {
          for (let index = 0; index < updates.length; index++) {
            batches[index]?.container.setEntityDestinations(updates[index].map(update => ({
              id: update.id,
              x: update.position[0],
              y: update.position[1],
              transition: false,
            })));
          }
        } else {
          optimizedRenderers[0]?.setPositions(updates[0]);
          optimizedRenderers[1]?.setPositions(updates[1]);
        }
      } catch (error) {
        console.error('[sprites-2d] motion worker step failed', error);
        animationEnabled = false;
      } finally {
        motionStepInFlight = false;
      }
    };
    scene.onBeforeRenderObservable.add(() => {
      const seconds = engine.getDeltaTime() * 0.001;
      for (const batch of batches) batch.container.tickTransitions(seconds);
      const now = performance.now();
      if (animationEnabled && useGpuMotion) {
        const dt = Math.min(0.25, Math.max(0, seconds));
        for (const renderer of optimizedRenderers) renderer.stepGpuMotion(now, dt);
      }
      if (animationEnabled && motionPool && !motionStepInFlight && now - previousMotionTime >= 50) {
        const dt = Math.min(0.2, (now - previousMotionTime) / 1000);
        previousMotionTime = now;
        void runMotionStep(now, dt);
      }
      for (const renderer of optimizedRenderers) {
        renderer.setViewFromOrthographicCamera(camera as any);
      }
      for (const renderer of textRenderers) renderer.setViewFromOrthographicCamera(camera as any);
      const textTick = Math.floor(now / 250);
      if (!loadTestEnabled && textTick !== previousTextTick) {
        previousTextTick = textTick;
        textRenderers[0]?.setText('text-live', `Runtime text ${textTick % 1000}`);
      }
    });

    if (loadTestEnabled) {
      let resolveResult!: (result: Sprite2DLoadTestResult) => void;
      const ready = new Promise<Sprite2DLoadTestResult>(resolve => { resolveResult = resolve; });
      (window as any).shadoSprite2DLoadTest = { ready, result: undefined };
      const frameDurations: number[] = [];
      const frameIntervals: number[] = [];
      const gpuDurations: number[] = [];
      let frameStarted = 0;
      let previousFrameEnded = 0;
      let firstFrameMs = 0;
      let renderedFrames = 0;
      const beforeObserver = scene.onBeforeRenderObservable.add(() => {
        frameStarted = performance.now();
      });
      const afterObserver = scene.onAfterRenderObservable.add(() => {
        const ended = performance.now();
        const duration = ended - frameStarted;
        renderedFrames++;
        if (renderedFrames === 1) firstFrameMs = duration;
        if (renderedFrames > loadWarmupFrames) {
          frameDurations.push(duration);
          if (previousFrameEnded > 0) frameIntervals.push(ended - previousFrameEnded);
          const gpuNanoseconds = loadInstrumentation?.gpuFrameTimeCounter.current ?? 0;
          if (gpuNanoseconds > 0) gpuDurations.push(gpuNanoseconds / 1_000_000);
        }
        previousFrameEnded = ended;
        if (frameDurations.length < loadSampleFrames) return;

        scene.onBeforeRenderObservable.remove(beforeObserver);
        scene.onAfterRenderObservable.remove(afterObserver);
        const optimizedStats = optimizedRenderers.map(renderer => renderer.getStats());
        const submittedSprites = useLegacyPath
          ? batches.reduce((sum, batch) => sum + batch.renderer.lastSubmittedInstances, 0)
          : optimizedStats.reduce((sum, value) => sum + value.visible, 0);
        const recordBytes = useLegacyPath ? 112 : 48;
        const intervalSummary = summarize(frameIntervals);
        const result: Sprite2DLoadTestResult = {
          path: useLegacyPath ? 'full' : 'optimized',
          backend: engine.isWebGPU ? 'webgpu' : 'webgl2',
          scenario: loadScenario,
          sprites: loadCount,
          submittedSprites,
          batches: useLegacyPath ? batches.length : optimizedRenderers.length,
          drawCalls: loadSceneInstrumentation?.drawCallsCounter.current ?? 0,
          recordBytes,
          workingSetBytes: loadCount * recordBytes,
          gpuCapacityBytes: useLegacyPath
            ? loadCount * recordBytes
            : optimizedStats.reduce((sum, value) => sum + value.gpuCapacityBytes, 0),
          setupMs,
          firstFrameMs,
          measuredFrames: frameDurations.length,
          frameMs: summarize(frameDurations),
          gpuMs: gpuDurations.length ? summarize(gpuDurations) : null,
          intervalMs: intervalSummary,
          effectiveFps: intervalSummary.mean > 0 ? 1000 / intervalSummary.mean : 0,
          gpuCulling: optimizedStats.some(value => value.gpuCullingActive),
          indirectDraw: optimizedStats.some(value => value.indirectDrawActive),
        };
        (window as any).shadoSprite2DLoadTest.result = result;
        resolveResult(result);
      });
    }

    const ui = GUI.AdvancedDynamicTexture.CreateFullscreenUI('sprites-2d-ui', true, scene);
    const panel = new GUI.StackPanel('sprites-2d-panel');
    panel.width = '330px';
    panel.left = '16px';
    panel.top = '58px';
    panel.horizontalAlignment = GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
    panel.verticalAlignment = GUI.Control.VERTICAL_ALIGNMENT_TOP;
    panel.background = '#071018dd';
    panel.paddingTop = '12px';
    panel.paddingLeft = '14px';
    panel.paddingRight = '14px';
    panel.paddingBottom = '12px';
    ui.addControl(panel);
    const title = new GUI.TextBlock('sprites-2d-title', 'Shado 2D rendering proof');
    title.height = '30px';
    title.color = '#ffffff';
    title.fontSize = 18;
    title.fontWeight = '600';
    title.textHorizontalAlignment = GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
    panel.addControl(title);
    const copy = new GUI.TextBlock(
      'sprites-2d-copy',
      `${useLegacyPath ? 'FULL BACKUP' : 'OPTIMIZED 2D'} · drag to pan · wheel to zoom\nOrientation stays locked to world Y: no orbit, pitch, or roll.\n${loadTestEnabled ? `LOAD TEST · ${loadCount.toLocaleString()} sprites · ${loadScenario}` : useLegacyPath ? 'Existing dynamic-entity renderer, unchanged fallback.' : '48-byte records · stable layers · tiled culling · zoom LOD.'}\nCutout and premultiplied batches remain live.`
    );
    copy.height = '112px';
    copy.color = '#b9cfdf';
    copy.fontSize = 13;
    copy.textWrapping = true;
    copy.textHorizontalAlignment = GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
    copy.textVerticalAlignment = GUI.Control.VERTICAL_ALIGNMENT_TOP;
    panel.addControl(copy);
    const pathButton = GUI.Button.CreateSimpleButton(
      'sprites-2d-path-toggle',
      useLegacyPath ? 'Use optimized 2D path' : 'Use legacy backup path'
    );
    pathButton.height = '34px';
    pathButton.color = '#ffffff';
    pathButton.background = useLegacyPath ? '#246f59' : '#27394b';
    pathButton.onPointerClickObservable.add(() => {
      const url = new URL(window.location.href);
      if (useLegacyPath) url.searchParams.delete('path');
      else url.searchParams.set('path', 'legacy');
      window.location.assign(url);
    });
    panel.addControl(pathButton);

    const stats = new GUI.TextBlock('sprites-2d-stats', '');
    stats.height = '44px';
    stats.color = '#8fd6ff';
    stats.fontSize = 12;
    stats.textHorizontalAlignment = GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
    panel.addControl(stats);
    pickReadout = new GUI.TextBlock(
      'sprites-2d-pick',
      useGpuMotion
        ? 'GPU motion: picking disabled to avoid position readback'
        : 'Click a sprite for exact 2D picking'
    );
    pickReadout.height = '26px';
    pickReadout.color = useLegacyPath ? '#8b9aa7' : '#ffd77a';
    pickReadout.fontSize = 12;
    pickReadout.textHorizontalAlignment = GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
    panel.addControl(pickReadout);
    const controlOverlay = loadTestEnabled ? undefined : createSprite2DControlOverlay(
      {
        path: useLegacyPath ? 'full' : 'optimized',
        population: activePopulationCount,
        layout: activeLayout,
        tileSize: configuredTileSize,
        lod: configuredLod,
        animation: animationEnabled,
        motionSeed,
        motionSpeed,
        motionCadence,
        motionWorkers: configuredMotionWorkers,
        gpuMotion: useGpuMotion,
        gpuCulling: gpuCullingEnabled,
        grid: ground.isEnabled(),
        text: 'ARBITRARY MSDF TEXT',
        fontSize: 1.15,
        align: 'center',
      },
      {
        spawn: replacePopulation,
        setLod(value) {
          for (const renderer of optimizedRenderers) renderer.setMinPixelSize(value);
          for (const renderer of textRenderers) renderer.setMinPixelSize(value);
          const url = new URL(window.location.href);
          url.searchParams.set('lod', String(value));
          window.history.replaceState(null, '', url);
        },
        setAnimation(value) { animationEnabled = value; },
        setMotion(seed, speed, cadence) {
          motionSeed = seed;
          motionSpeed = Math.max(0, Math.min(4, speed));
          motionCadence = cadence;
          const now = performance.now();
          void motionPool?.configure({
            seed: motionSeed,
            speed: motionSpeed,
            cadenceMs: motionCadence * 1000,
            bounds: motionBounds,
          }, now);
          if (useGpuMotion) {
            for (const renderer of optimizedRenderers) renderer.configureGpuMotion({
              seed: motionSeed,
              speed: motionSpeed,
              cadenceMs: motionCadence * 1000,
              bounds: motionBounds,
            });
          }
          previousMotionTime = now;
          const url = new URL(window.location.href);
          url.searchParams.set('motionSeed', String(motionSeed));
          url.searchParams.set('motionSpeed', String(motionSpeed));
          url.searchParams.set('motionCadence', String(motionCadence));
          window.history.replaceState(null, '', url);
        },
        setMotionWorkers(value) {
          const url = new URL(window.location.href);
          url.searchParams.set('motionWorkers', String(value));
          window.location.assign(url);
        },
        setGpuCulling(value) {
          const url = new URL(window.location.href);
          if (value) url.searchParams.set('gpuCulling', '1');
          else url.searchParams.delete('gpuCulling');
          window.location.assign(url);
        },
        setGrid(value) {
          ground.setEnabled(value);
          const url = new URL(window.location.href);
          if (value) url.searchParams.delete('grid');
          else url.searchParams.set('grid', '0');
          window.history.replaceState(null, '', url);
        },
        setText(text, fontSize, align) {
          textRenderers[0]?.upsert({
            id: 'text-title',
            text,
            position: [4, 7.1],
            fontSize,
            maxWidth: 14,
            align,
            color: [0.66, 0.91, 1, 1],
            layer: 8,
          });
        },
        resetView() {
          centerX = 0;
          centerZ = 0;
          zoomScale = 1;
          previousZoomScale = 0;
          updateCameraCenter();
          updateLockedProjection();
        },
        setPath(path) {
          const url = new URL(window.location.href);
          if (path === 'full') url.searchParams.set('path', 'legacy');
          else url.searchParams.delete('path');
          window.location.assign(url);
        },
        setTileSize(value) {
          const url = new URL(window.location.href);
          url.searchParams.set('tileSize', String(value));
          window.location.assign(url);
        },
      }
    );
    scene.onBeforeRenderObservable.add(() => {
      let visibleSprites = 0;
      let tileCount = 0;
      let gpuBytes = 0;
      let gpuMotionDispatches = 0;
      let gpuMotionError = '';
      let gpuCullingDispatches = 0;
      let indirectDraw = false;
      let gpuCullingError = '';
      if (useLegacyPath) {
        const total = batches.reduce((sum, batch) => sum + (batch.container.entityCount | 0), 0);
        const glyphs = textRenderers.reduce((sum, renderer) => sum + renderer.getStats().visibleGlyphs, 0);
        stats.text = `Legacy records: ${total} × 112 bytes\nMSDF text: ${glyphs} visible glyphs`;
        visibleSprites = total;
        gpuBytes = total * 112;
      } else {
        const values = optimizedRenderers.map(renderer => renderer.getStats());
        const glyphs = textRenderers.reduce((sum, renderer) => sum + renderer.getStats().visibleGlyphs, 0);
        stats.text = `Optimized: ${values.reduce((sum, value) => sum + value.visible, 0)}/${values.reduce((sum, value) => sum + value.total, 0)} visible · 48 bytes/record\nMSDF text: ${glyphs} visible glyphs`;
        visibleSprites = values.reduce((sum, value) => sum + value.visible, 0);
        tileCount = values.reduce((sum, value) => sum + value.tileCount, 0);
        gpuBytes = values.reduce((sum, value) => sum + value.gpuCapacityBytes, 0);
        gpuMotionDispatches = values.reduce((sum, value) => sum + value.gpuMotionDispatches, 0);
        gpuMotionError = values.find(value => value.gpuMotionError)?.gpuMotionError ?? '';
        gpuCullingDispatches = values.reduce((sum, value) => sum + value.gpuCullingDispatches, 0);
        indirectDraw = values.some(value => value.indirectDrawActive);
        gpuCullingError = values.find(value => value.gpuCullingError)?.gpuCullingError ?? '';
      }
      const textStats = textRenderers[0]?.getStats();
      controlOverlay?.updateStats(
        `${engine.getFps().toFixed(0)} FPS · ${engine.isWebGPU ? 'WebGPU' : 'WebGL2'}\n` +
        `${activePopulationCount.toLocaleString()} resident · ${visibleSprites.toLocaleString()} submitted\n` +
        `${activeLayout} · ${useGpuMotion ? gpuMotionError || gpuCullingError ? `GPU ERROR: ${(gpuMotionError || gpuCullingError).slice(0, 80)}` : `GPU motion ×${gpuMotionDispatches.toLocaleString()} · cull ×${gpuCullingDispatches.toLocaleString()}${indirectDraw ? ' indirect' : ''} · zero readback` : `${motionPool?.activeWorkers ?? 0}/${configuredMotionWorkers} SIMD workers`} · ${tileCount.toLocaleString()} tiles\n` +
        `${(gpuBytes / 1048576).toFixed(2)} MiB GPU records\n` +
        `${textStats?.visibleGlyphs ?? 0}/${textStats?.totalGlyphs ?? 0} glyphs · ${textStats?.unsupportedCharacters.length ?? 0} missing`
      );
    });

    scene.onDisposeObservable.add(() => {
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerUp);
      canvas.removeEventListener('wheel', onWheel);
      for (const batch of batches) batch.renderer.dispose();
      for (const renderer of optimizedRenderers) renderer.dispose();
      for (const renderer of textRenderers) renderer.dispose();
      loadInstrumentation?.dispose();
      loadSceneInstrumentation?.dispose();
      motionPool?.dispose();
      controlOverlay?.dispose();
      fontAsset?.dispose();
      atlas.dispose();
    });
    (window as any).shadoSprites2D = {
      scene, camera, atlas, batches, optimizedRenderers, textRenderers,
    };
    return scene;
  }
}
