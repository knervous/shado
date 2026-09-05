import {
  BABYLON,
  type AbstractEngine,
  type BaseTexture,
  type Buffer,
  type Camera,
  type Mesh,
  type Observer,
  type Scene,
  type ShaderMaterial,
} from '../babylon';
import type { ShadoSprite2DView } from './ShadoSprite2DRenderer';

const FLOATS_PER_GLYPH = 16;
const BYTES_PER_GLYPH = FLOATS_PER_GLYPH * 4;

export interface ShadoText2DFontAsset {
  textures: BaseTexture[];
  _font: {
    info: { size: number };
    common: { scaleW: number; scaleH: number; lineHeight: number };
    distanceField?: { distanceRange?: number };
    chars: Array<{
      id: number;
      x: number;
      y: number;
      width: number;
      height: number;
      xoffset: number;
      yoffset: number;
      xadvance: number;
    }>;
    kernings?: Array<{ first: number; second: number; amount: number }>;
  };
  _getKerning?(first: number, second: number): number;
}

export interface ShadoText2DInput {
  id: string;
  text: string;
  position: readonly [number, number];
  fontSize: number;
  color?: readonly [number, number, number, number];
  rotationRad?: number;
  rotationDeg?: number;
  visible?: boolean;
  layer?: number;
  order?: number;
  maxWidth?: number;
  lineHeight?: number;
  align?: 'left' | 'center' | 'right';
  pivot?: readonly [number, number];
  minPixelSize?: number;
}

export interface ShadoText2DRendererOptions {
  tileSize?: number;
  alphaCutoff?: number;
  thickness?: number;
  minPixelSize?: number;
}

export interface ShadoText2DPickResult {
  id: string;
  text: ShadoText2DInput;
  world: readonly [number, number];
  local: readonly [number, number];
}

export interface ShadoText2DStats {
  textBlocks: number;
  totalGlyphs: number;
  visibleGlyphs: number;
  tileCount: number;
  recordBytes: number;
  gpuCapacityBytes: number;
  drawListRebuilds: number;
  unsupportedCharacters: string[];
}

type FontGlyph = {
  codePoint: number;
  uv: readonly [number, number, number, number];
  plane: readonly [number, number, number, number];
  advance: number;
  drawable: boolean;
};

type LaidOutGlyph = {
  center: readonly [number, number];
  size: readonly [number, number];
  uv: readonly [number, number, number, number];
};

type TextRecord = {
  input: ShadoText2DInput;
  insertionOrder: number;
  glyphs: LaidOutGlyph[];
  bounds: readonly [number, number, number, number];
};

/** Arbitrary MSDF text rendered directly into locked orthographic 2D space. */
export class ShadoText2DRenderer {
  public readonly mesh: Mesh;
  public readonly material: ShaderMaterial;

  private readonly engine: AbstractEngine;
  private readonly records = new Map<string, TextRecord>();
  private readonly tiles = new Map<string, string[]>();
  private readonly glyphByCodePoint = new Map<number, FontGlyph>();
  private readonly kerning = new Map<string, number>();
  private readonly unsupported = new Set<string>();
  private readonly tileSize: number;
  private readonly alphaCutoff: number;
  private readonly thickness: number;
  private defaultMinPixelSize: number;
  private readonly beforeRenderObserver: Observer<Scene> | null;
  private instanceBuffer?: Buffer;
  private packed = new Float32Array(0);
  private capacity = 0;
  private insertionCounter = 0;
  private revision = 0;
  private signature = '';
  private maxRadius = 0;
  private totalGlyphs = 0;
  private visibleGlyphs = 0;
  private drawListRebuilds = 0;
  private view: ShadoSprite2DView = {
    center: [0, 0],
    halfExtent: [1, 1],
    viewportPixels: [1, 1],
  };

  public constructor(
    private readonly scene: Scene,
    private readonly font: ShadoText2DFontAsset,
    options: ShadoText2DRendererOptions = {}
  ) {
    if (!font.textures[0]) throw new Error('ShadoText2DRenderer requires an MSDF font texture');
    this.engine = scene.getEngine();
    this.tileSize = Math.max(0.25, options.tileSize ?? 8);
    this.alphaCutoff = Math.max(0, Math.min(1, options.alphaCutoff ?? 0.001));
    this.thickness = options.thickness ?? 0;
    this.defaultMinPixelSize = Math.max(0, options.minPixelSize ?? 0.75);
    this.buildFontLookup();
    this.mesh = this.createQuad();
    this.material = this.createMaterial();
    this.mesh.material = this.material;
    this.mesh.alwaysSelectAsActiveMesh = true;
    this.ensureCapacity(1);
    this.beforeRenderObserver = scene.onBeforeRenderObservable.add(() => {
      this.rebuildVisibleDrawList();
      this.material.setVector2('uCameraCenter', new BABYLON.Vector2(...this.view.center));
      this.material.setVector2('uCameraHalfExtent', new BABYLON.Vector2(...this.view.halfExtent));
      this.material.setFloat('uInstanceCount', Math.max(1, this.visibleGlyphs));
    });
  }

  public upsert(input: ShadoText2DInput): void {
    this.upsertMany([input]);
  }

  public upsertMany(inputs: readonly ShadoText2DInput[]): void {
    for (const input of inputs) {
      const current = this.records.get(input.id);
      const normalized = normalizeText(input);
      const layout = this.layout(normalized);
      this.records.set(input.id, {
        input: normalized,
        insertionOrder: current?.insertionOrder ?? this.insertionCounter++,
        ...layout,
      });
    }
    this.rebuildTiles();
  }

  public remove(id: string): boolean {
    if (!this.records.delete(id)) return false;
    this.rebuildTiles();
    return true;
  }

  public setMinPixelSize(value: number): void {
    const next = Math.max(0, value);
    if (Math.abs(next - this.defaultMinPixelSize) < 0.0001) return;
    this.defaultMinPixelSize = next;
    this.signature = '';
  }

  public setText(id: string, text: string): boolean {
    const record = this.records.get(id);
    if (!record || record.input.text === text) return false;
    this.upsert({ ...record.input, text });
    return true;
  }

  public setPosition(id: string, position: readonly [number, number]): boolean {
    const record = this.records.get(id);
    if (!record) return false;
    const previousTile = this.tileKey(record.input.position[0], record.input.position[1]);
    const nextTile = this.tileKey(position[0], position[1]);
    record.input = { ...record.input, position: [position[0], position[1]] };
    if (previousTile === nextTile) {
      this.revision++;
      this.signature = '';
    } else {
      this.rebuildTiles();
    }
    return true;
  }

  public setView(view: ShadoSprite2DView): void {
    this.view = {
      center: [view.center[0], view.center[1]],
      halfExtent: [Math.max(0.0001, view.halfExtent[0]), Math.max(0.0001, view.halfExtent[1])],
      viewportPixels: [Math.max(1, view.viewportPixels[0]), Math.max(1, view.viewportPixels[1])],
    };
  }

  public setViewFromOrthographicCamera(camera: Camera): void {
    const position = camera.globalPosition ?? camera.position;
    this.setView({
      center: [position.x, position.z],
      halfExtent: [
        Math.abs(((camera as any).orthoRight ?? 1) - ((camera as any).orthoLeft ?? -1)) * 0.5,
        Math.abs(((camera as any).orthoTop ?? 1) - ((camera as any).orthoBottom ?? -1)) * 0.5,
      ],
      viewportPixels: [this.engine.getRenderWidth(), this.engine.getRenderHeight()],
    });
  }

  public pickScreen(
    screenX: number,
    screenY: number,
    viewportWidth = this.view.viewportPixels[0],
    viewportHeight = this.view.viewportPixels[1]
  ): ShadoText2DPickResult | null {
    const worldX =
      this.view.center[0] + ((screenX / viewportWidth) * 2 - 1) * this.view.halfExtent[0];
    const worldY =
      this.view.center[1] + (1 - (screenY / viewportHeight) * 2) * this.view.halfExtent[1];
    const candidates = this.candidates(worldX, worldY)
      .map(id => this.records.get(id))
      .filter((record): record is TextRecord => !!record && record.input.visible !== false)
      .sort(compareRecords)
      .reverse();
    for (const record of candidates) {
      const dx = worldX - record.input.position[0];
      const dy = worldY - record.input.position[1];
      const rotation = -(record.input.rotationRad ?? 0);
      const c = Math.cos(rotation);
      const s = Math.sin(rotation);
      const localX = dx * c - dy * s;
      const localY = dx * s + dy * c;
      const [minX, minY, maxX, maxY] = record.bounds;
      if (localX < minX || localX > maxX || localY < minY || localY > maxY) continue;
      return {
        id: record.input.id,
        text: record.input,
        world: [worldX, worldY],
        local: [localX, localY],
      };
    }
    return null;
  }

  public getStats(): ShadoText2DStats {
    return {
      textBlocks: this.records.size,
      totalGlyphs: this.totalGlyphs,
      visibleGlyphs: this.visibleGlyphs,
      tileCount: this.tiles.size,
      recordBytes: BYTES_PER_GLYPH,
      gpuCapacityBytes: this.capacity * BYTES_PER_GLYPH,
      drawListRebuilds: this.drawListRebuilds,
      unsupportedCharacters: [...this.unsupported],
    };
  }

  public dispose(): void {
    if (this.beforeRenderObserver)
      this.scene.onBeforeRenderObservable.remove(this.beforeRenderObserver);
    this.mesh.forcedInstanceCount = 0;
    for (const kind of ['iTransform', 'iUvRect', 'iColor', 'iState']) {
      if (this.mesh.isVerticesDataPresent(kind)) this.mesh.removeVerticesData(kind);
    }
    this.instanceBuffer?.dispose();
    this.material.dispose();
    this.mesh.dispose();
  }

  private buildFontLookup(): void {
    const definition = this.font._font;
    const em = Math.max(1, Math.abs(definition.info.size));
    for (const glyph of definition.chars) {
      this.glyphByCodePoint.set(glyph.id, {
        codePoint: glyph.id,
        uv: [
          glyph.x / definition.common.scaleW,
          glyph.y / definition.common.scaleH,
          (glyph.x + glyph.width) / definition.common.scaleW,
          (glyph.y + glyph.height) / definition.common.scaleH,
        ],
        plane: [
          glyph.xoffset / em,
          -(glyph.yoffset + glyph.height) / em,
          (glyph.xoffset + glyph.width) / em,
          -glyph.yoffset / em,
        ],
        advance: glyph.xadvance / em,
        drawable: glyph.width > 0 && glyph.height > 0,
      });
    }
    for (const pair of definition.kernings ?? []) {
      this.kerning.set(`${pair.first}:${pair.second}`, pair.amount / em);
    }
  }

  private layout(input: ShadoText2DInput): Pick<TextRecord, 'glyphs' | 'bounds'> {
    const fontSize = input.fontSize;
    const em = Math.max(1, Math.abs(this.font._font.info.size));
    const defaultAdvance = 0.5;
    const lineHeight = input.lineHeight ?? (this.font._font.common.lineHeight / em) * fontSize;
    const maxWidth = Math.max(0, input.maxWidth ?? Number.POSITIVE_INFINITY);
    const fallback = this.glyphByCodePoint.get(0xfffc) ?? this.glyphByCodePoint.get(63);
    const lines: Array<{ glyphs: LaidOutGlyph[]; width: number }> = [];
    let glyphs: LaidOutGlyph[] = [];
    let penX = 0;
    let previous: number | undefined;
    let lineIndex = 0;
    const finishLine = () => {
      lines.push({ glyphs, width: penX });
      glyphs = [];
      penX = 0;
      previous = undefined;
      lineIndex++;
    };

    for (const character of input.text) {
      if (character === '\n') {
        finishLine();
        continue;
      }
      const codePoint = character.codePointAt(0)!;
      const source = this.glyphByCodePoint.get(codePoint);
      const glyph = source ?? fallback;
      if (!source && !/^\s$/u.test(character)) this.unsupported.add(character);
      const kerningPx =
        previous === undefined ? 0 : (this.font._getKerning?.(previous, codePoint) ?? 0);
      const kerning =
        previous === undefined
          ? 0
          : (this.kerning.get(`${previous}:${codePoint}`) ?? kerningPx / em);
      const advance = (glyph?.advance ?? defaultAdvance) * fontSize;
      if (penX > 0 && penX + kerning * fontSize + advance > maxWidth) finishLine();
      penX += kerning * fontSize;
      if (glyph?.drawable && !/^\s$/u.test(character)) {
        const [x0, y0, x1, y1] = glyph.plane;
        const left = penX + x0 * fontSize;
        const bottom = -lineIndex * lineHeight + y0 * fontSize;
        const right = penX + x1 * fontSize;
        const top = -lineIndex * lineHeight + y1 * fontSize;
        glyphs.push({
          center: [(left + right) * 0.5, (bottom + top) * 0.5],
          size: [right - left, top - bottom],
          uv: glyph.uv,
        });
      }
      penX += advance;
      previous = codePoint;
    }
    finishLine();

    const blockWidth = Number.isFinite(maxWidth)
      ? maxWidth
      : lines.reduce((width, line) => Math.max(width, line.width), 0);
    const laidOut: LaidOutGlyph[] = [];
    for (const line of lines) {
      const shift =
        input.align === 'right'
          ? blockWidth - line.width
          : input.align === 'center'
            ? (blockWidth - line.width) * 0.5
            : 0;
      for (const glyph of line.glyphs) {
        laidOut.push({ ...glyph, center: [glyph.center[0] + shift, glyph.center[1]] });
      }
    }

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const glyph of laidOut) {
      minX = Math.min(minX, glyph.center[0] - glyph.size[0] * 0.5);
      minY = Math.min(minY, glyph.center[1] - glyph.size[1] * 0.5);
      maxX = Math.max(maxX, glyph.center[0] + glyph.size[0] * 0.5);
      maxY = Math.max(maxY, glyph.center[1] + glyph.size[1] * 0.5);
    }
    if (!laidOut.length) minX = minY = maxX = maxY = 0;
    const pivot = input.pivot ?? [0.5, 0.5];
    const offsetX = -minX - (maxX - minX) * pivot[0];
    const offsetY = -minY - (maxY - minY) * pivot[1];
    for (const glyph of laidOut) {
      glyph.center = [glyph.center[0] + offsetX, glyph.center[1] + offsetY];
    }
    return {
      glyphs: laidOut,
      bounds: [minX + offsetX, minY + offsetY, maxX + offsetX, maxY + offsetY],
    };
  }

  private rebuildTiles(): void {
    this.tiles.clear();
    this.maxRadius = 0;
    this.totalGlyphs = 0;
    for (const [id, record] of this.records) {
      const key = this.tileKey(record.input.position[0], record.input.position[1]);
      let ids = this.tiles.get(key);
      if (!ids) this.tiles.set(key, (ids = []));
      ids.push(id);
      this.totalGlyphs += record.glyphs.length;
      const [minX, minY, maxX, maxY] = record.bounds;
      this.maxRadius = Math.max(this.maxRadius, Math.hypot(maxX - minX, maxY - minY) * 0.5);
    }
    for (const ids of this.tiles.values()) {
      ids.sort((a, b) => compareRecords(this.records.get(a)!, this.records.get(b)!));
    }
    this.revision++;
    this.signature = '';
  }

  private rebuildVisibleDrawList(): void {
    const minX = this.view.center[0] - this.view.halfExtent[0] - this.maxRadius;
    const maxX = this.view.center[0] + this.view.halfExtent[0] + this.maxRadius;
    const minY = this.view.center[1] - this.view.halfExtent[1] - this.maxRadius;
    const maxY = this.view.center[1] + this.view.halfExtent[1] + this.maxRadius;
    const bounds = this.tileBounds(minX, minY, maxX, maxY);
    const pixelsPerUnit = this.view.viewportPixels[1] / (this.view.halfExtent[1] * 2);
    const lodBucket = Math.round(Math.log2(Math.max(0.0001, pixelsPerUnit)) * 8);
    const signature = `${bounds.join(':')}:${lodBucket}:${this.revision}`;
    if (signature === this.signature) return;
    this.signature = signature;
    const visible = this.candidatesForBounds(bounds)
      .map(id => this.records.get(id))
      .filter(
        (record): record is TextRecord =>
          !!record &&
          record.input.visible !== false &&
          record.input.fontSize * pixelsPerUnit >=
            (record.input.minPixelSize ?? this.defaultMinPixelSize)
      )
      .sort(compareRecords);
    const glyphCount = visible.reduce((sum, record) => sum + record.glyphs.length, 0);
    this.ensureCapacity(Math.max(1, glyphCount));
    this.packed.fill(0);
    let index = 0;
    for (const record of visible) {
      for (const glyph of record.glyphs) this.packGlyph(index++, record, glyph);
    }
    this.instanceBuffer!.update(this.packed);
    this.visibleGlyphs = glyphCount;
    this.mesh.forcedInstanceCount = glyphCount;
    this.mesh.isVisible = glyphCount > 0;
    this.drawListRebuilds++;
  }

  private packGlyph(index: number, record: TextRecord, glyph: LaidOutGlyph): void {
    const offset = index * FLOATS_PER_GLYPH;
    const rotation = record.input.rotationRad ?? 0;
    const c = Math.cos(rotation);
    const s = Math.sin(rotation);
    const localX = glyph.center[0];
    const localY = glyph.center[1];
    this.packed[offset] = record.input.position[0] + localX * c - localY * s;
    this.packed[offset + 1] = record.input.position[1] + localX * s + localY * c;
    this.packed[offset + 2] = glyph.size[0];
    this.packed[offset + 3] = glyph.size[1];
    this.packed.set(glyph.uv, offset + 4);
    this.packed.set(record.input.color ?? [1, 1, 1, 1], offset + 8);
    this.packed[offset + 12] = rotation;
    this.packed[offset + 13] = Math.max(0, record.input.layer ?? 0);
  }

  private ensureCapacity(required: number): void {
    if (required <= this.capacity) return;
    let capacity = Math.max(16, this.capacity);
    while (capacity < required) capacity *= 2;
    for (const kind of ['iTransform', 'iUvRect', 'iColor', 'iState']) {
      if (this.mesh.isVerticesDataPresent(kind)) this.mesh.removeVerticesData(kind);
    }
    this.instanceBuffer?.dispose();
    this.capacity = capacity;
    this.packed = new Float32Array(capacity * FLOATS_PER_GLYPH);
    this.instanceBuffer = new BABYLON.Buffer(
      this.engine,
      this.packed,
      true,
      FLOATS_PER_GLYPH,
      false,
      true,
      false,
      1,
      'ShadoText2D compact glyph records'
    );
    this.mesh.setVerticesBuffer(
      this.instanceBuffer.createVertexBuffer('iTransform', 0, 4, FLOATS_PER_GLYPH, true)
    );
    this.mesh.setVerticesBuffer(
      this.instanceBuffer.createVertexBuffer('iUvRect', 4, 4, FLOATS_PER_GLYPH, true)
    );
    this.mesh.setVerticesBuffer(
      this.instanceBuffer.createVertexBuffer('iColor', 8, 4, FLOATS_PER_GLYPH, true)
    );
    this.mesh.setVerticesBuffer(
      this.instanceBuffer.createVertexBuffer('iState', 12, 4, FLOATS_PER_GLYPH, true)
    );
  }

  private candidates(x: number, y: number): string[] {
    return this.candidatesForBounds(
      this.tileBounds(
        x - this.maxRadius,
        y - this.maxRadius,
        x + this.maxRadius,
        y + this.maxRadius
      )
    );
  }

  private candidatesForBounds(bounds: readonly number[]): string[] {
    const result: string[] = [];
    const seen = new Set<string>();
    const columns = bounds[2] - bounds[0] + 1;
    const rows = bounds[3] - bounds[1] + 1;
    if (!Number.isFinite(columns * rows) || columns * rows > this.tiles.size * 4) {
      for (const [key, tile] of this.tiles) {
        const separator = key.indexOf(':');
        const x = Number(key.slice(0, separator));
        const y = Number(key.slice(separator + 1));
        if (x < bounds[0] || x > bounds[2] || y < bounds[1] || y > bounds[3]) continue;
        for (const id of tile) {
          if (!seen.has(id)) {
            seen.add(id);
            result.push(id);
          }
        }
      }
      return result;
    }
    for (let y = bounds[1]; y <= bounds[3]; y++) {
      for (let x = bounds[0]; x <= bounds[2]; x++) {
        for (const id of this.tiles.get(`${x}:${y}`) ?? []) {
          if (!seen.has(id)) {
            seen.add(id);
            result.push(id);
          }
        }
      }
    }
    return result;
  }

  private tileBounds(
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): [number, number, number, number] {
    return [
      Math.floor(minX / this.tileSize),
      Math.floor(minY / this.tileSize),
      Math.floor(maxX / this.tileSize),
      Math.floor(maxY / this.tileSize),
    ];
  }

  private tileKey(x: number, y: number): string {
    return `${Math.floor(x / this.tileSize)}:${Math.floor(y / this.tileSize)}`;
  }

  private createQuad(): Mesh {
    const mesh = new BABYLON.Mesh('shado-text-2d', this.scene);
    const data = new BABYLON.VertexData();
    data.positions = [-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0];
    data.uvs = [0, 1, 1, 1, 1, 0, 0, 0];
    data.indices = [0, 1, 2, 0, 2, 3];
    data.applyToMesh(mesh);
    return mesh;
  }

  private createMaterial(): ShaderMaterial {
    const webgpu = this.engine.isWebGPU;
    const shaderName = webgpu ? 'shadoText2DMSDFWGSL' : 'shadoText2DMSDFGLSL';
    installShaders(shaderName, webgpu);
    const material = new BABYLON.ShaderMaterial('shadoText2DMSDFMaterial', this.scene, shaderName, {
      attributes: ['position', 'uv', 'iTransform', 'iUvRect', 'iColor', 'iState'],
      uniforms: [
        'uCameraCenter',
        'uCameraHalfExtent',
        'uInstanceCount',
        'uAlphaCutoff',
        'uThickness',
      ],
      samplers: ['uFontAtlas'],
      uniformBuffers: ['Scene'],
      needAlphaBlending: true,
      shaderLanguage: webgpu ? BABYLON.ShaderLanguage.WGSL : BABYLON.ShaderLanguage.GLSL,
    });
    material.backFaceCulling = false;
    material.disableDepthWrite = true;
    material.alphaMode = BABYLON.Engine.ALPHA_COMBINE;
    material.setTexture('uFontAtlas', this.font.textures[0]);
    material.setFloat('uAlphaCutoff', this.alphaCutoff);
    material.setFloat('uThickness', this.thickness);
    return material;
  }
}

function normalizeText(input: ShadoText2DInput): ShadoText2DInput {
  return {
    ...input,
    position: [input.position[0], input.position[1]],
    fontSize: Math.max(0.0001, input.fontSize),
    color: input.color ? ([...input.color] as [number, number, number, number]) : [1, 1, 1, 1],
    rotationRad: input.rotationRad ?? ((input.rotationDeg ?? 0) * Math.PI) / 180,
    layer: Math.round(input.layer ?? 0),
    order: Math.round(input.order ?? 0),
    pivot: input.pivot ? ([...input.pivot] as [number, number]) : [0.5, 0.5],
  };
}

function compareRecords(a: TextRecord, b: TextRecord): number {
  return (
    (a.input.layer ?? 0) - (b.input.layer ?? 0) ||
    (a.input.order ?? 0) - (b.input.order ?? 0) ||
    a.insertionOrder - b.insertionOrder
  );
}

function installShaders(name: string, webgpu: boolean): void {
  if (webgpu) {
    BABYLON.ShaderStore.ShadersStoreWGSL[`${name}VertexShader`] ??= `
attribute position: vec3f;
attribute uv: vec2f;
attribute iTransform: vec4f;
attribute iUvRect: vec4f;
attribute iColor: vec4f;
attribute iState: vec4f;
uniform uCameraCenter: vec2f;
uniform uCameraHalfExtent: vec2f;
uniform uInstanceCount: f32;
varying vUV: vec2f;
varying vColor: vec4f;
@vertex
fn main(input: VertexInputs) -> FragmentInputs {
  let c = cos(vertexInputs.iState.x);
  let s = sin(vertexInputs.iState.x);
  let local = vertexInputs.position.xy * vertexInputs.iTransform.zw;
  let world = vertexInputs.iTransform.xy + vec2f(local.x*c-local.y*s, local.x*s+local.y*c);
  let clip = (world - uniforms.uCameraCenter) / uniforms.uCameraHalfExtent;
  let depth = 0.0009 - min(vertexInputs.iState.y, 255.0) * 0.000001 -
    (f32(vertexInputs.instanceIndex) / max(1.0, uniforms.uInstanceCount)) * 0.0000005;
  vertexOutputs.position = vec4f(clip, depth, 1.0);
  vertexOutputs.vUV = vec2f(
    mix(vertexInputs.iUvRect.x, vertexInputs.iUvRect.z, vertexInputs.uv.x),
    mix(vertexInputs.iUvRect.y, vertexInputs.iUvRect.w, vertexInputs.uv.y)
  );
  vertexOutputs.vColor = vertexInputs.iColor;
}`;
    BABYLON.ShaderStore.ShadersStoreWGSL[`${name}FragmentShader`] ??= `
varying vUV: vec2f;
varying vColor: vec4f;
uniform uAlphaCutoff: f32;
uniform uThickness: f32;
var uFontAtlasSampler: sampler;
var uFontAtlas: texture_2d<f32>;
fn median3(a: f32, b: f32, c: f32) -> f32 { return max(min(a,b), min(max(a,b),c)); }
@fragment
fn main(input: FragmentInputs) -> FragmentOutputs {
  let sample = textureSample(uFontAtlas, uFontAtlasSampler, fragmentInputs.vUV);
  let sd = median3(sample.r, sample.g, sample.b) - 0.5 + uniforms.uThickness;
  let alpha = clamp(sd / max(fwidth(sd), 0.0001) + 0.5, 0.0, 1.0) * sample.a * fragmentInputs.vColor.a;
  if (alpha <= uniforms.uAlphaCutoff) { discard; }
  fragmentOutputs.color = vec4f(fragmentInputs.vColor.rgb, alpha);
}`;
    return;
  }
  BABYLON.Effect.ShadersStore[`${name}VertexShader`] ??= `
precision highp float;
attribute vec3 position; attribute vec2 uv; attribute vec4 iTransform; attribute vec4 iUvRect;
attribute vec4 iColor; attribute vec4 iState;
uniform vec2 uCameraCenter; uniform vec2 uCameraHalfExtent; uniform float uInstanceCount;
varying vec2 vUV; varying vec4 vColor;
void main(void) {
  float c=cos(iState.x), s=sin(iState.x); vec2 local=position.xy*iTransform.zw;
  vec2 world=iTransform.xy+vec2(local.x*c-local.y*s,local.x*s+local.y*c);
  float depth=-0.999-min(iState.y,255.0)*0.000001-(float(gl_InstanceID)/max(1.0,uInstanceCount))*0.0000005;
  gl_Position=vec4((world-uCameraCenter)/uCameraHalfExtent,depth,1.0);
  vUV=vec2(mix(iUvRect.x,iUvRect.z,uv.x),mix(iUvRect.y,iUvRect.w,uv.y)); vColor=iColor;
}`;
  BABYLON.Effect.ShadersStore[`${name}FragmentShader`] ??= `
precision highp float;
uniform sampler2D uFontAtlas; uniform float uAlphaCutoff; uniform float uThickness;
varying vec2 vUV; varying vec4 vColor;
float median3(float a,float b,float c){return max(min(a,b),min(max(a,b),c));}
void main(void){vec4 sample=texture2D(uFontAtlas,vUV);float sd=median3(sample.r,sample.g,sample.b)-0.5+uThickness;
float alpha=clamp(sd/max(fwidth(sd),0.0001)+0.5,0.0,1.0)*sample.a*vColor.a;
if(alpha<=uAlphaCutoff)discard;gl_FragColor=vec4(vColor.rgb,alpha);}
`;
}
