import {
  appendSurfaceHeightField,
  compileCoverage,
  COVERAGE_RESOLUTION,
  SURFACE_HEIGHT_RESOLUTION,
} from './grass-coverage';
import type {
  ShadoWorldGrassFieldCompileOptions,
  ShadoWorldGrassFieldPackage,
  ShadoWorldPrimitive,
  ShadoWorldSpatialPackage,
} from './types';

type ResolvedFieldSettings = Required<ShadoWorldGrassFieldCompileOptions>;

const DEFAULTS = {
  cellSize: 24,
  density: 12,
  minimumUpNormal: 0.58,
  minHeight: 0.32,
  maxHeight: 0.68,
  bladeWidth: 0.052,
  seed: 0x0e17_a91a,
} satisfies ResolvedFieldSettings;

/**
 * Compiles tagged surfaces into a grass *field* rather than a blade list.
 *
 * The v1 package in `./grass` stores one record per blade, so its size grows
 * with density and it needs a hard placement cap to stay loadable. On a zone
 * whose floor is one large terrain primitive that cap is reached immediately:
 * Talios Crownward requested roughly 4.8M blades over 4.15 km² and shipped
 * 18,000, or 0.0045 blades/m².
 *
 * This package instead stores only where grass *may* grow — the coverage
 * raster and ground heights that v1 already computed — and leaves the blades
 * themselves to be derived at runtime from a hash of cell and blade index. Its
 * size is therefore a function of the grass *area*, and density becomes a
 * runtime quality knob that costs nothing to raise.
 */
export function compileShadoWorldGrassField(
  primitives: readonly ShadoWorldPrimitive[],
  options: ShadoWorldGrassFieldCompileOptions | false | undefined,
  blockerPrimitives: readonly ShadoWorldPrimitive[] = []
): ShadoWorldGrassFieldPackage | undefined {
  if (options === false || options === undefined) return undefined;
  const settings = { ...DEFAULTS, ...options };
  validateSettings(settings);

  const coverage = compileCoverage(primitives, settings, blockerPrimitives);
  const cells = [...coverage.values()].sort((a, b) => a.z - b.z || a.x - b.x);

  const output: ShadoWorldGrassFieldPackage = {
    version: 2,
    cellSize: settings.cellSize,
    density: settings.density,
    minHeight: settings.minHeight,
    maxHeight: settings.maxHeight,
    bladeWidth: settings.bladeWidth,
    seed: settings.seed,
    cells: { x: [], z: [] },
    coverage: {
      resolution: COVERAGE_RESOLUTION,
      wordsPerCell: COVERAGE_RESOLUTION,
      words: [],
    },
    heightField: {
      resolution: SURFACE_HEIGHT_RESOLUTION,
      wordsPerCell: Math.ceil((SURFACE_HEIGHT_RESOLUTION * SURFACE_HEIGHT_RESOLUTION) / 32),
      words: [],
      minimumY: [],
      heightRange: [],
      samples: [],
    },
  };

  for (const cell of cells) {
    output.cells.x.push(cell.x);
    output.cells.z.push(cell.z);
    appendSurfaceHeightField(output.heightField, cell.heights);
    for (let word = 0; word < COVERAGE_RESOLUTION; word++) {
      let bits = 0;
      for (let bit = 0; bit < 32; bit++) {
        if (Number.isFinite(cell.heights[word * 32 + bit])) {
          bits = (bits | (1 << bit)) >>> 0;
        }
      }
      output.coverage.words.push(bits);
    }
  }
  return output;
}

/** Blades a cell is worth at a given density, before any runtime LOD. */
export function shadoWorldGrassBladesPerCell(
  field: Pick<ShadoWorldGrassFieldPackage, 'cellSize' | 'density'>
): number {
  return Math.round(field.cellSize * field.cellSize * field.density);
}

/**
 * Fraction of a cell that actually bears grass, from its coverage bits.
 *
 * A cell on the edge of a lawn is mostly empty, and generating a full blade
 * quota there would spend most of it on blades that collapse immediately.
 */
export function shadoWorldGrassCellCoverage(
  field: ShadoWorldGrassFieldPackage,
  cell: number
): number {
  const wordsPerCell = field.coverage.wordsPerCell;
  const first = cell * wordsPerCell;
  let covered = 0;
  for (let word = 0; word < wordsPerCell; word++) {
    let bits = field.coverage.words[first + word]!;
    // Kernighan's popcount: grass rasters are sparse at the edges, so this
    // iterates once per set bit rather than 32 times per word.
    while (bits) {
      bits &= bits - 1;
      covered++;
    }
  }
  return covered / (field.coverage.resolution * field.coverage.resolution);
}

function validateSettings(settings: ResolvedFieldSettings): void {
  const positive = [
    ['cellSize', settings.cellSize],
    ['density', settings.density],
    ['maxHeight', settings.maxHeight],
    ['bladeWidth', settings.bladeWidth],
  ] as const;
  for (const [name, value] of positive) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`grassField.${name} must be positive`);
    }
  }
  if (
    !Number.isFinite(settings.minHeight) ||
    settings.minHeight <= 0 ||
    settings.minHeight > settings.maxHeight
  ) {
    throw new Error('grassField.minHeight must be positive and not exceed maxHeight');
  }
  if (
    !Number.isFinite(settings.minimumUpNormal) ||
    settings.minimumUpNormal < 0 ||
    settings.minimumUpNormal > 1
  ) {
    throw new Error('grassField.minimumUpNormal must be between zero and one');
  }
}


/**
 * Reads a v2 field out of a package that only carries the v1 blade grass.
 *
 * The two formats describe the same surface from the same rasterization; v1
 * simply also stored a blade list and then capped it. Zones baked before the
 * field existed therefore already contain everything the field needs, so this
 * lets the runtime render them without a re-bake. Prefer `world.grassField`
 * when it is present.
 */
export function shadoWorldGrassFieldFromPackage(
  world: Pick<ShadoWorldSpatialPackage, 'grass' | 'grassField'>,
  overrides: Pick<
    ShadoWorldGrassFieldCompileOptions,
    'density' | 'minHeight' | 'maxHeight' | 'bladeWidth' | 'seed'
  > = {}
): ShadoWorldGrassFieldPackage | undefined {
  if (world.grassField) return world.grassField;
  const grass = world.grass;
  const heightField = grass?.coverage?.heightField;
  if (!grass || !grass.coverage || !heightField) return undefined;
  return {
    version: 2,
    cellSize: grass.cellSize,
    density: overrides.density ?? DEFAULTS.density,
    minHeight: overrides.minHeight ?? DEFAULTS.minHeight,
    maxHeight: overrides.maxHeight ?? DEFAULTS.maxHeight,
    bladeWidth: overrides.bladeWidth ?? DEFAULTS.bladeWidth,
    seed: overrides.seed ?? DEFAULTS.seed,
    cells: { x: grass.cells.x, z: grass.cells.z },
    coverage: {
      resolution: grass.coverage.resolution,
      wordsPerCell: grass.coverage.wordsPerCell,
      words: grass.coverage.words,
    },
    heightField,
  };
}
