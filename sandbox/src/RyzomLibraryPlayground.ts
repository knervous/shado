import * as BABYLON from '@babylonjs/core';
import '@babylonjs/loaders/glTF';

/**
 * Ryzom asset library: browse everything the intake converted, in one pane.
 *
 * The point is triage speed — step through props, weapons, buildings, creatures
 * and actors, see each one framed immediately, and mark what is worth promoting
 * into Eltania. Decisions persist to assets/ryzom/candidates.json through the
 * dev-server plugin, which is the same ledger the static review page uses.
 *
 * Assets stream from the repo rather than public/: the library is ~5.7 GB and
 * copying it into the sandbox would be absurd.
 */

type CatalogEntry = {
  id: string;
  name: string;
  category: string;
  source: string;
  triangles: number;
  meshes: number;
  materials: number;
  textures: number;
  animations: number;
  skinned: boolean;
  joints?: number;
  bytes: number;
  size: [number, number, number] | null;
  warnings: number;
  unresolvedTextures: boolean;
  reviewStatus: string;
  eltaniaMapping: string | null;
  notes: string[];
  promotedId: string | null;
  promotedAt: string | null;
  sourceFamily: string | null;
  licenseConfidence: string | null;
  /** Non-null when provenance bars promotion; the promote endpoint enforces it. */
  promotionBlocked: string | null;
  styleReview: string | null;
  provenanceOverride: string | null;
};

type Catalog = {
  total: number;
  byCategory: Record<string, number>;
  entries: CatalogEntry[];
};

const REVIEW_STATES = [
  'unreviewed',
  'candidate',
  'eltania-ready',
  'needs-remodel',
  'needs-retexture',
  'reject',
] as const;

const STATE_COLOR: Record<string, string> = {
  'unreviewed': '#7c8794',
  'candidate': '#7cc4ff',
  'eltania-ready': '#7ee08a',
  'needs-remodel': '#e0a458',
  'needs-retexture': '#d99ae0',
  'reject': '#e06c6c',
};

export class RyzomLibraryPlayground {
  public static async CreateScene(
    engine: BABYLON.Engine,
    canvas: HTMLCanvasElement
  ): Promise<BABYLON.Scene> {
    const scene = new BABYLON.Scene(engine);
    scene.clearColor = BABYLON.Color4.FromHexString('#0e1116ff');
    (globalThis as any).__shadoScene = scene;

    const camera = new BABYLON.ArcRotateCamera(
      'ryzom-library-camera',
      -Math.PI / 2,
      1.15,
      6,
      BABYLON.Vector3.Zero(),
      scene
    );
    camera.attachControl(canvas, true);
    camera.wheelPrecision = 40;
    camera.panningSensibility = 60;
    camera.lowerRadiusLimit = 0.05;
    camera.upperRadiusLimit = 5000;

    const key = new BABYLON.HemisphericLight('key', new BABYLON.Vector3(0.3, 1, 0.2), scene);
    key.intensity = 1.0;
    const rim = new BABYLON.DirectionalLight('rim', new BABYLON.Vector3(-0.5, -1, 0.4), scene);
    rim.intensity = 0.8;

    const catalog = await fetch('/__ryzom/catalog')
      .then((r) => (r.ok ? (r.json() as Promise<Catalog>) : null))
      .catch(() => null);

    if (!catalog?.entries.length) {
      mountHint(canvas);
      return scene;
    }

    const ui = mountUi(canvas, catalog);
    let container: BABYLON.AssetContainer | null = null;
    let token = 0;

    async function show(entry: CatalogEntry): Promise<void> {
      const mine = ++token;
      ui.setLoading(entry);
      container?.dispose();
      container = null;

      const url = `/__ryzom/asset/${entry.id}.glb`;
      try {
        const loaded = await BABYLON.LoadAssetContainerAsync(url, scene, { pluginExtension: '.glb' });
        // A newer click landed while this was in flight; drop this result.
        if (mine !== token) { loaded.dispose(); return; }
        loaded.addAllToScene();
        container = loaded;

        frame(loaded, camera, scene);
        const clips = loaded.animationGroups.map((group) => group.name);
        // Actors carry hundreds of clips; play one so the pane is not a T-pose.
        for (const group of loaded.animationGroups) group.stop();
        loaded.animationGroups[0]?.play(true);
        ui.setLoaded(entry, clips, (name) => {
          for (const group of loaded.animationGroups) group.stop();
          loaded.animationGroups.find((g) => g.name === name)?.play(true);
        });
      } catch (error) {
        if (mine === token) ui.setError(entry, error instanceof Error ? error.message : String(error));
      }
    }

    ui.onSelect(show);
    scene.onDisposeObservable.add(() => {
      container?.dispose();
      ui.dispose();
    });
    return scene;
  }
}

/** Frame the loaded asset regardless of its scale — a weapon and a keep both fit. */
function frame(
  container: BABYLON.AssetContainer,
  camera: BABYLON.ArcRotateCamera,
  scene: BABYLON.Scene
): void {
  let min: BABYLON.Vector3 | null = null;
  let max: BABYLON.Vector3 | null = null;
  for (const mesh of container.meshes) {
    if (!mesh.getTotalVertices()) continue;
    mesh.computeWorldMatrix(true);
    const box = mesh.getBoundingInfo().boundingBox;
    min = min ? BABYLON.Vector3.Minimize(min, box.minimumWorld) : box.minimumWorld.clone();
    max = max ? BABYLON.Vector3.Maximize(max, box.maximumWorld) : box.maximumWorld.clone();
  }
  if (!min || !max) return;
  const center = BABYLON.Vector3.Center(min, max);
  const extent = max.subtract(min);
  const size = Math.max(extent.x, extent.y, extent.z) || 1;
  camera.target.copyFrom(center);
  camera.radius = size * 2.1;
  camera.lowerRadiusLimit = size * 0.05;
  camera.upperRadiusLimit = size * 40;
  camera.alpha = -Math.PI / 2;
  camera.beta = 1.15;
  scene.render();
}

type UiHandle = {
  onSelect: (handler: (entry: CatalogEntry) => void) => void;
  setLoading: (entry: CatalogEntry) => void;
  setLoaded: (entry: CatalogEntry, clips: string[], play: (name: string) => void) => void;
  setError: (entry: CatalogEntry, message: string) => void;
  dispose: () => void;
};

function mountUi(canvas: HTMLCanvasElement, catalog: Catalog): UiHandle {
  const parent = canvas.parentElement ?? document.body;
  if (getComputedStyle(parent).position === 'static') parent.style.position = 'relative';

  const root = document.createElement('aside');
  root.style.cssText = [
    'position:absolute', 'inset:12px auto 12px 12px', 'z-index:30',
    'width:340px', 'max-width:calc(100vw - 24px)', 'display:flex', 'flex-direction:column',
    'gap:8px', 'background:#141a22ee', 'border:1px solid #26303c', 'border-radius:10px',
    'padding:10px', 'color:#e6eaf0', 'font:12px/1.5 system-ui', 'backdrop-filter:blur(6px)',
  ].join(';');

  const cats = Object.keys(catalog.byCategory).sort();
  let activeCategory = cats.includes('prop') ? 'prop' : cats[0]!;
  let query = '';
  /** all | promoted | unpromoted — the sweep is mostly "what have I not done yet". */
  let promotionFilter: 'all' | 'promoted' | 'unpromoted' = 'all';
  let selected: CatalogEntry | null = null;
  let handler: ((entry: CatalogEntry) => void) | null = null;

  root.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:baseline">
      <strong style="font:650 13px system-ui">Ryzom library</strong>
      <span style="color:#8c98a6">${catalog.total.toLocaleString()} assets</span>
    </div>
    <div data-cats style="display:flex;flex-wrap:wrap;gap:4px"></div>
    <input data-search placeholder="filter by name…" style="background:#0e141b;border:1px solid #26303c;border-radius:6px;padding:6px 8px;color:inherit;font:inherit">
    <div data-promo style="display:flex;gap:4px"></div>
    <div data-count style="color:#8c98a6"></div>
    <div data-list style="flex:1;min-height:0;overflow:auto;border-top:1px solid #26303c;padding-top:6px"></div>
    <div data-detail style="border-top:1px solid #26303c;padding-top:8px"></div>
  `;
  parent.appendChild(root);

  const catsEl = root.querySelector('[data-cats]') as HTMLElement;
  const searchEl = root.querySelector('[data-search]') as HTMLInputElement;
  const promoEl = root.querySelector('[data-promo]') as HTMLElement;
  const countEl = root.querySelector('[data-count]') as HTMLElement;
  const listEl = root.querySelector('[data-list]') as HTMLElement;
  const detailEl = root.querySelector('[data-detail]') as HTMLElement;

  function visible(): CatalogEntry[] {
    const q = query.toLowerCase();
    return catalog.entries.filter((e) => {
      if (e.category !== activeCategory) return false;
      if (q && !e.name.toLowerCase().includes(q)) return false;
      if (promotionFilter === 'promoted' && !e.promotedId) return false;
      if (promotionFilter === 'unpromoted' && e.promotedId) return false;
      return true;
    });
  }

  function promotedCount(category: string): number {
    return catalog.entries.filter((e) => e.category === category && e.promotedId).length;
  }

  function renderCats(): void {
    catsEl.innerHTML = cats
      .map((c) => {
        const on = c === activeCategory;
        return `<button data-cat="${c}" style="background:${on ? '#1d2a38' : '#0e141b'};border:1px solid ${on ? '#3d5871' : '#26303c'};color:${on ? '#7cc4ff' : '#aab4c0'};border-radius:99px;padding:3px 9px;font:600 11px system-ui;cursor:pointer">${c} ${catalog.byCategory[c]}</button>`;
      })
      .join('');
    for (const button of catsEl.querySelectorAll('button')) {
      button.addEventListener('click', () => {
        activeCategory = (button as HTMLElement).dataset.cat!;
        renderCats();
        renderList();
      });
    }
  }

  function renderPromoFilter(): void {
    const options: Array<[typeof promotionFilter, string]> = [
      ['all', 'all'], ['unpromoted', 'not promoted'], ['promoted', 'promoted'],
    ];
    promoEl.innerHTML = options
      .map(([value, label]) => {
        const on = promotionFilter === value;
        return `<button data-promo-filter="${value}" style="flex:1;background:${on ? '#1d2a38' : '#0e141b'};border:1px solid ${on ? '#3d5871' : '#26303c'};color:${on ? '#7cc4ff' : '#8c98a6'};border-radius:5px;padding:3px 0;font:600 10px system-ui;cursor:pointer">${label}</button>`;
      })
      .join('');
    for (const button of promoEl.querySelectorAll('[data-promo-filter]')) {
      button.addEventListener('click', () => {
        promotionFilter = (button as HTMLElement).dataset.promoFilter as typeof promotionFilter;
        renderPromoFilter();
        renderList();
      });
    }
  }

  function renderList(): void {
    const rows = visible();
    const promoted = promotedCount(activeCategory);
    countEl.textContent = `${rows.length} shown · ${promoted} promoted in ${activeCategory}`;
    listEl.innerHTML = rows
      .map((e, i) => {
        const dot = STATE_COLOR[e.reviewStatus] ?? '#7c8794';
        const on = selected?.id === e.id;
        // A promoted asset is the one thing worth spotting without reading, so
        // it gets a mark of its own rather than another shade of status dot.
        const mark = e.promotedId
          ? '<span title="promoted to Eltania" style="color:#7ee08a;font-size:11px;flex:none">&#10003;</span>'
          : '';
        return `<div data-i="${i}" style="display:flex;align-items:center;gap:6px;padding:4px 6px;border-radius:5px;cursor:pointer;background:${on ? '#1d2a38' : 'transparent'}">
          <span style="width:7px;height:7px;border-radius:99px;background:${dot};flex:none"></span>
          <span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${e.name}</span>
          ${mark}
          <span style="color:#66707c;font-size:11px">${e.triangles.toLocaleString()}</span>
        </div>`;
      })
      .join('');
    for (const row of listEl.querySelectorAll('[data-i]')) {
      row.addEventListener('click', () => {
        selected = rows[Number((row as HTMLElement).dataset.i)]!;
        renderList();
        handler?.(selected);
      });
    }
  }

  function renderDetail(entry: CatalogEntry | null, body: string): void {
    if (!entry) { detailEl.innerHTML = '<span style="color:#8c98a6">Select an asset.</span>'; return; }
    const buttons = REVIEW_STATES.map(
      (state) => `<button data-state="${state}" style="background:${entry.reviewStatus === state ? '#1d2a38' : '#0e141b'};border:1px solid ${entry.reviewStatus === state ? STATE_COLOR[state] : '#26303c'};color:${entry.reviewStatus === state ? STATE_COLOR[state] : '#aab4c0'};border-radius:5px;padding:3px 7px;font:600 10px system-ui;cursor:pointer">${state}</button>`
    ).join('');
    const promoted = entry.promotedId
      ? `<div style="color:#7ee08a;margin-bottom:6px">&#10003; promoted as <code>${entry.promotedId}</code></div>`
      : '';
    // Where an asset came from, and whether that is settled. Core art shows
    // nothing here; it has no family and was never in question.
    const provenance = entry.sourceFamily
      ? `<div style="margin-bottom:6px;font-size:11px">
          <span style="color:#8c98a6">${entry.sourceFamily}</span>
          <span style="color:${entry.licenseConfidence === 'green' ? '#7ee08a' : '#e0b755'}"> · ${entry.licenseConfidence ?? 'unknown'}</span>
          ${entry.styleReview ? '<span style="color:#e0b755"> · style review</span>' : ''}
        </div>`
      : '';
    // An override is not the same as verified provenance, and the pane should
    // never let the two look alike.
    const override = entry.provenanceOverride
      ? `<div style="margin-bottom:6px;padding:5px 7px;border:1px solid #4a4326;background:#231f13;border-radius:6px;color:#c9b26a;font-size:11px">
          Promotable by owner decision, not by established licence.
          <div style="color:#8f8a66;margin-top:3px">${entry.provenanceOverride}</div>
        </div>`
      : '';
    const blocked = entry.promotionBlocked
      ? `<div style="margin-bottom:6px;padding:5px 7px;border:1px solid #5c4a22;background:#2a2113;border-radius:6px;color:#e0b755;font-size:11px">
          Blocked from promotion — ${entry.promotionBlocked}. A geometry match to a
          licensed source clears it; run <code>fingerprint_dedupe --mode provenance</code>.
        </div>`
      : '';
    detailEl.innerHTML = `
      <div style="font:650 12px system-ui;margin-bottom:2px">${entry.name}</div>
      <div style="color:#66707c;font-size:11px;word-break:break-all;margin-bottom:6px">${entry.source}</div>
      <div style="color:#aab4c0;margin-bottom:6px">${body}</div>
      ${provenance}
      ${promoted}
      ${override}
      ${blocked}
      <button data-promote ${entry.promotionBlocked ? 'disabled' : ''} style="width:100%;margin-bottom:6px;background:${entry.promotionBlocked ? '#171b20' : entry.promotedId ? '#16241b' : '#1b3324'};border:1px solid ${entry.promotionBlocked ? '#2a3138' : entry.promotedId ? '#2f4a38' : '#3f7a55'};color:${entry.promotionBlocked ? '#5b646d' : '#7ee08a'};border-radius:6px;padding:6px;font:650 11px system-ui;cursor:${entry.promotionBlocked ? 'not-allowed' : 'pointer'}">
        ${entry.promotionBlocked ? 'Promotion blocked' : entry.promotedId ? 'Re-promote to Eltania object' : 'Promote to Eltania object'}
      </button>
      <div data-promote-status style="color:#8c98a6;margin-bottom:6px"></div>
      <div style="display:flex;flex-wrap:wrap;gap:4px">${buttons}</div>
    `;
    const promoteButton = detailEl.querySelector('[data-promote]') as HTMLButtonElement | null;
    const promoteStatus = detailEl.querySelector('[data-promote-status]') as HTMLElement | null;
    promoteButton?.addEventListener('click', async () => {
      promoteButton.disabled = true;
      if (promoteStatus) promoteStatus.textContent = 'promoting…';
      try {
        const result = await fetch('/__ryzom/promote', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: entry.id }),
        }).then((r) => r.json());
        if (result.error) throw new Error(result.error);
        entry.promotedId = result.objectId;
        entry.promotedAt = new Date().toISOString();
        entry.reviewStatus = 'eltania-ready';
        renderList();
        renderDetail(entry, body);
      } catch (error) {
        promoteButton.disabled = false;
        if (promoteStatus) {
          promoteStatus.innerHTML = `<span style="color:#e06c6c">${error instanceof Error ? error.message : String(error)}</span>`;
        }
      }
    });
    for (const button of detailEl.querySelectorAll('[data-state]')) {
      button.addEventListener('click', async () => {
        const state = (button as HTMLElement).dataset.state!;
        entry.reviewStatus = state;
        renderList();
        renderDetail(entry, body);
        await fetch('/__ryzom/review', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source: entry.source, reviewStatus: state }),
        }).catch(() => undefined);
      });
    }
  }

  searchEl.addEventListener('input', () => { query = searchEl.value; renderList(); });
  renderPromoFilter();
  renderCats();
  renderList();
  renderDetail(null, '');

  function stats(entry: CatalogEntry): string {
    const dims = entry.size ? `${entry.size.join(' × ')}` : '—';
    return [
      `${entry.triangles.toLocaleString()} tris · ${entry.meshes} mesh · ${entry.materials} mat · ${entry.textures} tex`,
      `size ${dims}`,
      entry.animations ? `${entry.animations} clips · ${entry.joints ?? 0} joints` : 'static',
      `${(entry.bytes / 2 ** 20).toFixed(2)} MB${entry.unresolvedTextures ? ' · missing textures' : ''}`,
    ].join('<br>');
  }

  return {
    onSelect(next) { handler = next; },
    setLoading(entry) { renderDetail(entry, 'loading…'); },
    setLoaded(entry, clips, play) {
      renderDetail(entry, stats(entry));
      if (clips.length > 1) {
        const picker = document.createElement('select');
        picker.style.cssText = 'width:100%;margin-top:6px;background:#0e141b;border:1px solid #26303c;border-radius:5px;padding:4px;color:inherit;font:inherit';
        picker.innerHTML = clips.map((c) => `<option>${c}</option>`).join('');
        picker.addEventListener('change', () => play(picker.value));
        detailEl.appendChild(picker);
      }
    },
    setError(entry, message) { renderDetail(entry, `<span style="color:#e06c6c">${message}</span>`); },
    dispose() { root.remove(); },
  };
}

function mountHint(canvas: HTMLCanvasElement): void {
  const parent = canvas.parentElement ?? document.body;
  if (getComputedStyle(parent).position === 'static') parent.style.position = 'relative';
  const hint = document.createElement('div');
  hint.style.cssText = 'position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;color:#cfd8e3;font:500 13px/1.6 system-ui;text-align:center';
  hint.innerHTML = `
    <div style="font:650 15px system-ui">No converted Ryzom assets found</div>
    <div>Build the library first, from the repo root:</div>
    <code style="background:#0e141b;border:1px solid #26303c;border-radius:6px;padding:8px 12px;color:#7cc4ff">
      npm run ryzom:convert -- --exclude terrain,animation<br>npm run ryzom:actors
    </code>
  `;
  parent.appendChild(hint);
}
