/**
 * Controls and the proof overlay for the hum wardrobe module-draw page.
 *
 * The overlay exists to make one claim falsifiable on screen: instance count
 * and draw count are independent. Scale the crowd and watch `draws` stay flat
 * while `actors` climbs.
 */

export type HumWardrobeControls = {
  model: string;
  models: string[];
  setModel(model: string): void;
  /** Shown in the subtitle, so it has to come from the manifest, not a literal. */
  submeshCount: number;
  moduleCount: number;
  pieceCount: number;
  pieces: { piece: string; label: string; variations: string[] }[];
  clips: string[];
  setCount(count: number, variability: number): void;
  setVariability(variability: number): void;
  setPieceVariation(piece: string, variationIndex: number | null): void;
  setPieceTint(piece: string, color: { r: number; g: number; b: number }): void;
  /** A distinct dye per actor per piece, re-rolled on each call. */
  randomizeTints(): void;
  resetTints(): void;
  setClip(name: string): void;
};

export type HumWardrobeReport = {
  actors: number;
  drawCalls: number;
  moduleDraws: number;
  moduleTotal: number;
  submittedVertices: number;
  baselineVertices: number;
  vertexWorkReduction: number;
  fps: number;
  frameMs: number;
};

const COUNTS = [1, 64, 256, 1000, 2500, 5000, 10000];

const CSS = `
.hw-panel{position:fixed;z-index:9;top:14px;right:14px;width:min(400px,calc(100vw - 28px));max-height:calc(100vh - 28px);overflow:auto;padding:14px;border:1px solid #2b3a4a;border-radius:12px;background:#080d13ee;color:#dfe8f2;text-align:left;box-shadow:0 18px 60px #000a;backdrop-filter:blur(14px);font:12px/1.45 system-ui,sans-serif}
.hw-panel h1{margin:0 0 2px;font-size:16px}
.hw-panel h2{margin:14px 0 6px;color:#8fa4b8;font-size:9px;letter-spacing:.12em;text-transform:uppercase}
.hw-panel p{margin:0 0 8px;color:#8494a6;font-size:11px}
.hw-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.hw-stat{padding:7px 8px;border:1px solid #23303d;border-radius:7px;background:#0e151d}
.hw-stat b{display:block;color:#eaf3fb;font:14px/1.2 ui-monospace,monospace}
.hw-stat span{color:#6d7f92;font-size:8.5px;letter-spacing:.07em;text-transform:uppercase}
.hw-stat[data-key="draws"] b{color:#6fd98a}
.hw-stat[data-key="actors"] b{color:#7fb8ff}
.hw-row{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:7px}
.hw-row button,.hw-panel select,.hw-panel input[type=number]{height:26px;padding:0 8px;border:1px solid #2d3f51;border-radius:6px;background:#111a24;color:#d6e3f0;font-size:11px}
.hw-row button{cursor:pointer}
.hw-row button:hover{border-color:#5aa9e6}
.hw-row button[data-active="true"]{border-color:#6fd98a;color:#8ef0a6}
.hw-field{display:grid;grid-template-columns:52px 1fr 30px;align-items:center;gap:6px;margin-bottom:5px}
.hw-field label{color:#7f92a6;font-size:10px}
.hw-field select{width:100%;min-width:0}
.hw-field input[type=color]{width:30px;height:24px;padding:0;border:1px solid #2d3f51;border-radius:5px;background:#111a24;cursor:pointer}
.hw-slider{display:grid;grid-template-columns:1fr 44px;align-items:center;gap:8px;margin-bottom:8px}
.hw-slider input{width:100%}
.hw-note{margin-top:10px;color:#61748a;font:10px/1.5 ui-monospace,monospace}
.hw-note b{color:#8fa4b8;font-weight:400}
@media(max-width:640px){.hw-panel{width:calc(100vw - 20px);right:10px;top:10px;max-height:52vh}}
`;

export function createHumWardrobeUi(controls: HumWardrobeControls) {
  const style = document.createElement('style');
  style.textContent = CSS;
  document.head.append(style);

  const panel = document.createElement('section');
  panel.className = 'hw-panel';
  panel.innerHTML = `
    <h1 data-role="title">wardrobe &mdash; module draws</h1>
    <p>Ryzom-derived body: ${controls.submeshCount} submeshes in
       ${controls.moduleCount} (piece, variation) modules across
       ${controls.pieceCount} pieces. Each module draws only the actors
       wearing it.</p>

    <h2>Proof</h2>
    <div class="hw-grid">
      <div class="hw-stat" data-key="actors"><b data-role="actors">0</b><span>Actors</span></div>
      <div class="hw-stat" data-key="draws"><b data-role="draws">0</b><span>Draw calls</span></div>
      <div class="hw-stat"><b data-role="modules">0</b><span>Modules drawn</span></div>
      <div class="hw-stat"><b data-role="fps">0</b><span>FPS</span></div>
      <div class="hw-stat"><b data-role="frame">0</b><span>Frame ms</span></div>
      <div class="hw-stat"><b data-role="reduction">1.00&times;</b><span>Vertex saving</span></div>
    </div>
    <div class="hw-note" data-role="verts"></div>

    <h2>Body</h2>
    <div class="hw-row" data-role="models"></div>

    <h2>Crowd</h2>
    <div class="hw-row" data-role="counts"></div>
    <div class="hw-slider">
      <input data-role="variability" type="range" min="0" max="100" value="65">
      <span data-role="variability-value">65%</span>
    </div>
    <p style="margin:-4px 0 0">Outfit variability: 0% dresses everyone alike, which
       collapses the crowd onto the fewest modules.</p>

    <h2>Outfit</h2>
    <div class="hw-row">
      <button data-role="randomize-tints">Randomize tints</button>
      <button data-role="reset-tints">Reset</button>
    </div>
    <div data-role="pieces"></div>

    <h2>Animation</h2>
    <div class="hw-row"><select data-role="clip"></select></div>
  `;
  document.body.append(panel);

  const role = <T extends HTMLElement>(name: string) =>
    panel.querySelector<T>(`[data-role="${name}"]`)!;

  let variability = 0.65;
  let count = 256;

  const modelsHost = role('models');
  for (const model of controls.models) {
    const button = document.createElement('button');
    button.textContent = model;
    button.dataset.active = String(model === controls.model);
    // Swapping the body swaps geometry, atlas and VAT together, so this
    // reloads rather than trying to hot-swap three containers at once.
    button.addEventListener('click', () => controls.setModel(model));
    modelsHost.append(button);
  }

  const countsHost = role('counts');
  const countButtons: HTMLButtonElement[] = [];
  for (const preset of COUNTS) {
    const button = document.createElement('button');
    button.textContent = preset.toLocaleString();
    button.dataset.active = String(preset === count);
    button.addEventListener('click', () => {
      count = preset;
      for (const other of countButtons) other.dataset.active = String(other === button);
      controls.setCount(count, variability);
    });
    countsHost.append(button);
    countButtons.push(button);
  }

  const variabilityInput = role<HTMLInputElement>('variability');
  const variabilityValue = role('variability-value');
  variabilityInput.addEventListener('input', () => {
    variability = Number(variabilityInput.value) / 100;
    variabilityValue.textContent = `${variabilityInput.value}%`;
  });
  variabilityInput.addEventListener('change', () => controls.setVariability(variability));

  const piecesHost = role('pieces');
  for (const piece of controls.pieces) {
    const field = document.createElement('div');
    field.className = 'hw-field';
    const label = document.createElement('label');
    label.textContent = piece.label;
    const select = document.createElement('select');
    // A piece with one variation still gets a row, so the wardrobe reads as
    // what it is rather than looking like pieces went missing.
    const random = document.createElement('option');
    random.value = 'random';
    random.textContent = piece.variations.length > 1 ? 'mixed' : piece.variations[0];
    select.append(random);
    for (const [index, variation] of piece.variations.entries()) {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = variation;
      select.append(option);
    }
    select.addEventListener('change', () => {
      controls.setPieceVariation(
        piece.piece,
        select.value === 'random' ? null : Number(select.value)
      );
    });
    const colour = document.createElement('input');
    colour.type = 'color';
    colour.value = '#ffffff';
    colour.addEventListener('input', () => {
      const hex = colour.value;
      controls.setPieceTint(piece.piece, {
        r: parseInt(hex.slice(1, 3), 16) / 255,
        g: parseInt(hex.slice(3, 5), 16) / 255,
        b: parseInt(hex.slice(5, 7), 16) / 255,
      });
    });
    field.append(label, select, colour);
    piecesHost.append(field);
  }

  role('randomize-tints').addEventListener('click', () => controls.randomizeTints());
  role('reset-tints').addEventListener('click', () => {
    controls.resetTints();
    for (const swatch of panel.querySelectorAll<HTMLInputElement>('input[type=color]')) {
      swatch.value = '#ffffff';
    }
  });

  const clipSelect = role<HTMLSelectElement>('clip');
  for (const clip of controls.clips) {
    const option = document.createElement('option');
    option.value = clip;
    option.textContent = clip;
    if (clip === 'mixed') option.selected = true;
    clipSelect.append(option);
  }
  clipSelect.addEventListener('change', () => controls.setClip(clipSelect.value));

  const actors = role('actors');
  const draws = role('draws');
  const modules = role('modules');
  const fps = role('fps');
  const frame = role('frame');
  const reduction = role('reduction');
  const verts = role('verts');

  return {
    report(report: HumWardrobeReport) {
      actors.textContent = report.actors.toLocaleString();
      draws.textContent = report.drawCalls.toLocaleString();
      modules.textContent = `${report.moduleDraws}/${report.moduleTotal}`;
      fps.textContent = report.fps.toFixed(0);
      frame.textContent = report.frameMs.toFixed(1);
      reduction.textContent = `${report.vertexWorkReduction.toFixed(2)}×`;
      verts.textContent =
        `submitted <b>${report.submittedVertices.toLocaleString()}</b> verts vs ` +
        `<b>${report.baselineVertices.toLocaleString()}</b> for one merged supermesh`;
      verts.innerHTML = verts.textContent;
    },
    dispose() {
      panel.remove();
      style.remove();
    },
  };
}
