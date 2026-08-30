import {
  appendSurfaceHeightField,
  compileCoverage,
  COVERAGE_RESOLUTION,
  eligibleTriangles,
  randomGenerator,
  SURFACE_HEIGHT_RESOLUTION,
  type CoverageCell,
  type SurfaceTriangle,
} from './grass-coverage';
import type {
  ShadoWorldGrassCompileOptions,
  ShadoWorldGrassPackage,
  ShadoWorldPrimitive,
} from './types';

type Placement = {
  x: number;
  y: number;
  z: number;
  yaw: number;
  width: number;
  height: number;
  phase: number;
  stiffness: number;
  colorVariation: number;
};

type ResolvedGrassSettings = Required<ShadoWorldGrassCompileOptions>;

const DEFAULTS = {
  cellSize: 24,
  density: 1.15,
  maxPlacements: 48_000,
  maxPlacementsPerPrimitive: 18_000,
  minimumUpNormal: 0.58,
  minHeight: 0.32,
  maxHeight: 0.68,
  bladeWidth: 0.052,
  seed: 0x0e17_a91a,
} satisfies ResolvedGrassSettings;

/** Converts tagged source surfaces into deterministic, cell-owned placements. */
export function compileShadoWorldGrass(
  primitives: readonly ShadoWorldPrimitive[],
  options: ShadoWorldGrassCompileOptions | false | undefined,
  blockerPrimitives: readonly ShadoWorldPrimitive[] = []
): ShadoWorldGrassPackage | undefined {
  if (options === false) return undefined;
  const settings = { ...DEFAULTS, ...options };
  validateSettings(settings);
  const sources = primitives
    .map((primitive, index) => ({ primitive, index }))
    .filter(({ primitive }) => primitive.extraShader === 'grass');
  // An explicit grass policy is also a runtime package contract. Preserve an
  // empty versioned package when the authored zone has no tagged surfaces so
  // loaders and audits do not have to infer whether grass was intentionally
  // compiled or accidentally skipped. The default `undefined` policy still
  // omits grass for worlds that did not request it.
  if (!sources.length) {
    return options ? bucketPlacements([], settings, new Map()) : undefined;
  }
  const coverage = compileCoverage(primitives, settings, blockerPrimitives);

  const placements: Placement[] = [];
  let remaining = settings.maxPlacements;
  for (let source = 0; source < sources.length && remaining > 0; source++) {
    const { primitive, index } = sources[source]!;
    const triangles = eligibleTriangles(primitive, settings.minimumUpNormal);
    if (!triangles.length) continue;
    const totalArea = triangles[triangles.length - 1]!.cumulativeArea;
    const sourceLimit = Math.min(
      settings.maxPlacementsPerPrimitive,
      Math.ceil(remaining / (sources.length - source))
    );
    const count = Math.min(sourceLimit, Math.max(1, Math.round(totalArea * settings.density)));
    placements.push(
      ...sampleSurface(
        primitive,
        triangles,
        totalArea,
        count,
        settings,
        (settings.seed + Math.imul(index, 0x9e37_79b1)) >>> 0
      )
    );
    remaining -= count;
  }
  if (!placements.length) return bucketPlacements([], settings, coverage);
  return bucketPlacements(placements, settings, coverage);
}

function sampleSurface(
  primitive: ShadoWorldPrimitive,
  triangles: readonly SurfaceTriangle[],
  totalArea: number,
  count: number,
  settings: ResolvedGrassSettings,
  seed: number
): Placement[] {
  const random = randomGenerator(seed);
  const positions = primitive.positions;
  const output: Placement[] = [];
  for (let placement = 0; placement < count; placement++) {
    const targetArea = random() * totalArea;
    let low = 0;
    let high = triangles.length - 1;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (triangles[middle]!.cumulativeArea < targetArea) low = middle + 1;
      else high = middle;
    }
    const triangle = triangles[low]!;
    const rootU = Math.sqrt(random());
    const baryA = 1 - rootU;
    const baryB = rootU * (1 - random());
    const baryC = 1 - baryA - baryB;
    const interpolate = (axis: number) =>
      Number(positions[triangle.a + axis]) * baryA +
      Number(positions[triangle.b + axis]) * baryB +
      Number(positions[triangle.c + axis]) * baryC;
    output.push({
      x: interpolate(0),
      y: interpolate(1) + 0.012,
      z: interpolate(2),
      yaw: random() * Math.PI,
      width: settings.bladeWidth * 2 * (0.72 + random() * 0.52),
      height: settings.minHeight + (settings.maxHeight - settings.minHeight) * random(),
      phase: random(),
      stiffness: random(),
      colorVariation: random(),
    });
  }
  return output;
}

function bucketPlacements(
  placements: readonly Placement[],
  settings: ResolvedGrassSettings,
  coverage: ReadonlyMap<string, CoverageCell>
): ShadoWorldGrassPackage {
  const cellSize = settings.cellSize;
  const buckets = new Map<string, Placement[]>();
  for (const placement of placements) {
    const x = Math.floor(placement.x / cellSize);
    const z = Math.floor(placement.z / cellSize);
    const key = `${x}:${z}`;
    const coverageCell = coverage.get(key);
    const localX = Math.min(
      COVERAGE_RESOLUTION - 1,
      Math.max(0, Math.floor(((placement.x - x * cellSize) / cellSize) * COVERAGE_RESOLUTION))
    );
    const localZ = Math.min(
      COVERAGE_RESOLUTION - 1,
      Math.max(0, Math.floor(((placement.z - z * cellSize) / cellSize) * COVERAGE_RESOLUTION))
    );
    if (!Number.isFinite(coverageCell?.heights[localZ * COVERAGE_RESOLUTION + localX])) continue;
    const bucket = buckets.get(key);
    if (bucket) bucket.push(placement);
    else buckets.set(key, [placement]);
  }
  const cells = [...coverage.values()]
    .map(cell => {
      const key = `${cell.x}:${cell.z}`;
      const values = buckets.get(key) ?? [];
      if (!values.length) {
        const sample = cell.heights.findIndex(Number.isFinite);
        const localX = sample % COVERAGE_RESOLUTION;
        const localZ = Math.floor(sample / COVERAGE_RESOLUTION);
        const random = randomGenerator(
          (Math.imul(cell.x, 73_856_093) ^ Math.imul(cell.z, 19_349_663) ^ settings.seed) >>> 0
        );
        values.push({
          x: cell.x * cellSize + ((localX + 0.5) / COVERAGE_RESOLUTION) * cellSize,
          y: cell.heights[sample]! + 0.012,
          z: cell.z * cellSize + ((localZ + 0.5) / COVERAGE_RESOLUTION) * cellSize,
          yaw: random() * Math.PI,
          width: settings.bladeWidth * 2 * (0.72 + random() * 0.52),
          height: settings.minHeight + (settings.maxHeight - settings.minHeight) * random(),
          phase: random(),
          stiffness: random(),
          colorVariation: random(),
        });
      }
      return { ...cell, values };
    })
    .sort((a, b) => a.z - b.z || a.x - b.x);
  const output: ShadoWorldGrassPackage = {
    version: 1,
    cellSize,
    cells: {
      x: [],
      z: [],
      firstPlacement: [],
      placementCount: [],
    },
    placements: {
      positionX: [],
      positionY: [],
      positionZ: [],
      yaw: [],
      width: [],
      height: [],
      phase: [],
      stiffness: [],
      colorVariation: [],
    },
    coverage: {
      resolution: COVERAGE_RESOLUTION,
      wordsPerCell: COVERAGE_RESOLUTION,
      words: [],
      heightField: {
        resolution: SURFACE_HEIGHT_RESOLUTION,
        wordsPerCell: Math.ceil((SURFACE_HEIGHT_RESOLUTION * SURFACE_HEIGHT_RESOLUTION) / 32),
        words: [],
        minimumY: [],
        heightRange: [],
        samples: [],
      },
    },
  };
  for (const cell of cells) {
    output.cells.x.push(cell.x);
    output.cells.z.push(cell.z);
    output.cells.firstPlacement.push(output.placements.positionX.length);
    output.cells.placementCount.push(cell.values.length);
    appendSurfaceHeightField(output.coverage!.heightField!, cell.heights);
    for (let word = 0; word < COVERAGE_RESOLUTION; word++) {
      let bits = 0;
      for (let bit = 0; bit < 32; bit++) {
        if (Number.isFinite(cell.heights[word * 32 + bit])) {
          bits = (bits | (1 << bit)) >>> 0;
        }
      }
      output.coverage!.words.push(bits);
    }
    for (const placement of cell.values) {
      output.placements.positionX.push(placement.x);
      output.placements.positionY.push(placement.y);
      output.placements.positionZ.push(placement.z);
      output.placements.yaw.push(placement.yaw);
      output.placements.width.push(placement.width);
      output.placements.height.push(placement.height);
      output.placements.phase.push(placement.phase);
      output.placements.stiffness.push(placement.stiffness);
      output.placements.colorVariation.push(placement.colorVariation);
    }
  }
  return output;
}

function validateSettings(settings: ResolvedGrassSettings): void {
  const positive = [
    ['cellSize', settings.cellSize],
    ['density', settings.density],
    ['maxPlacements', settings.maxPlacements],
    ['maxPlacementsPerPrimitive', settings.maxPlacementsPerPrimitive],
    ['maxHeight', settings.maxHeight],
    ['bladeWidth', settings.bladeWidth],
  ] as const;
  for (const [name, value] of positive) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`grass.${name} must be positive`);
    }
  }
  if (
    !Number.isFinite(settings.minHeight) ||
    settings.minHeight <= 0 ||
    settings.minHeight > settings.maxHeight
  ) {
    throw new Error('grass.minHeight must be positive and not exceed maxHeight');
  }
  if (
    !Number.isFinite(settings.minimumUpNormal) ||
    settings.minimumUpNormal < 0 ||
    settings.minimumUpNormal > 1
  ) {
    throw new Error('grass.minimumUpNormal must be between zero and one');
  }
  if (
    !Number.isInteger(settings.maxPlacements) ||
    !Number.isInteger(settings.maxPlacementsPerPrimitive)
  ) {
    throw new Error('grass placement limits must be integers');
  }
}
