import type { ShadoWorldGrassPackage, ShadoWorldPrimitive } from './types';

export type SurfaceTriangle = {
  a: number;
  b: number;
  c: number;
  cumulativeArea: number;
};

/** Per-cell raster of the highest grass-bearing surface at each texel. */
export type CoverageCell = {
  x: number;
  z: number;
  heights: Float64Array;
};

export const COVERAGE_RESOLUTION = 32;
export const SURFACE_HEIGHT_RESOLUTION = 8;
export const SURFACE_HEIGHT_QUANTIZATION_MAX = 0xffff;
const BLOCKER_BELOW_TOLERANCE = 0.12;

export type CoverageSettings = {
  cellSize: number;
  minimumUpNormal: number;
};

export function compileCoverage(
  primitives: readonly ShadoWorldPrimitive[],
  settings: CoverageSettings,
  blockerPrimitives: readonly ShadoWorldPrimitive[]
): Map<string, CoverageCell> {
  const cells = new Map<string, CoverageCell>();
  for (const primitive of primitives) {
    if (primitive.extraShader !== 'grass') continue;
    const triangles = eligibleTriangles(primitive, settings.minimumUpNormal);
    for (const triangle of triangles) {
      rasterizeTriangle(primitive, triangle, settings.cellSize, (x, z, sample, y) => {
        const key = `${x}:${z}`;
        let cell = cells.get(key);
        if (!cell) {
          const heights = new Float64Array(COVERAGE_RESOLUTION * COVERAGE_RESOLUTION);
          heights.fill(Number.NaN);
          cell = { x, z, heights };
          cells.set(key, cell);
        }
        const previous = cell.heights[sample]!;
        if (!Number.isFinite(previous) || y > previous) cell.heights[sample] = y;
      });
    }
  }

  // Any upward-facing authored geometry at or above the terrain suppresses the
  // grass below it. This deliberately includes elevated floors, bridges,
  // walkways, and roofs: grass blades must never poke through a structure just
  // because its walking surface is more than a couple of metres above ground.
  // Foliage is unaffected because water/sky are skipped and near-vertical
  // surfaces do not pass the up-normal threshold.
  const blockedCells = new Set<CoverageCell & { blocked?: Uint8Array }>();
  for (const primitive of [...primitives, ...blockerPrimitives]) {
    if (
      primitive.extraShader === 'grass' ||
      primitive.extraShader === 'water' ||
      primitive.extraShader === 'sky'
    ) {
      continue;
    }
    const triangles = eligibleTriangles(primitive, 0.35);
    for (const triangle of triangles) {
      rasterizeTriangle(primitive, triangle, settings.cellSize, (x, z, sample, y) => {
        const cell = cells.get(`${x}:${z}`) as (CoverageCell & { blocked?: Uint8Array }) | undefined;
        const grassY = cell?.heights[sample];
        if (
          cell &&
          Number.isFinite(grassY) &&
          y >= grassY! - BLOCKER_BELOW_TOLERANCE
        ) {
          cell.heights[sample] = Number.NaN;
          if (!cell.blocked) {
            cell.blocked = new Uint8Array(COVERAGE_RESOLUTION * COVERAGE_RESOLUTION);
            blockedCells.add(cell);
          }
          cell.blocked[sample] = 1;
        }
      });
    }
  }
  // Triangle coverage is sampled at texel centres. Dilate one texel so blades
  // whose roots land just inside a structural edge cannot survive because the
  // nearest blocker sample centre happened to fall just outside it.
  for (const cell of blockedCells) {
    const blocked = cell.blocked!;
    for (let sample = 0; sample < blocked.length; sample++) {
      if (!blocked[sample]) continue;
      const localX = sample % COVERAGE_RESOLUTION;
      const localZ = Math.floor(sample / COVERAGE_RESOLUTION);
      const gridX = cell.x * COVERAGE_RESOLUTION + localX;
      const gridZ = cell.z * COVERAGE_RESOLUTION + localZ;
      for (let dz = -1; dz <= 1; dz++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dz === 0) continue;
          const sampleX = gridX + dx;
          const sampleZ = gridZ + dz;
          const targetCellX = Math.floor(sampleX / COVERAGE_RESOLUTION);
          const targetCellZ = Math.floor(sampleZ / COVERAGE_RESOLUTION);
          const targetLocalX = ((sampleX % COVERAGE_RESOLUTION) + COVERAGE_RESOLUTION) % COVERAGE_RESOLUTION;
          const targetLocalZ = ((sampleZ % COVERAGE_RESOLUTION) + COVERAGE_RESOLUTION) % COVERAGE_RESOLUTION;
          const targetCell = targetCellX === cell.x && targetCellZ === cell.z ? cell : cells.get(`${targetCellX}:${targetCellZ}`);
          if (targetCell) targetCell.heights[targetLocalZ * COVERAGE_RESOLUTION + targetLocalX] = Number.NaN;
        }
      }
    }
    delete cell.blocked;
  }
  for (const [key, cell] of cells) {
    if (!cell.heights.some(Number.isFinite)) cells.delete(key);
  }
  return cells;
}

export function rasterizeTriangle(
  primitive: ShadoWorldPrimitive,
  triangle: SurfaceTriangle,
  cellSize: number,
  visit: (cellX: number, cellZ: number, sample: number, y: number) => void
): void {
  const positions = primitive.positions;
  const ax = Number(positions[triangle.a]);
  const ay = Number(positions[triangle.a + 1]);
  const az = Number(positions[triangle.a + 2]);
  const bx = Number(positions[triangle.b]);
  const by = Number(positions[triangle.b + 1]);
  const bz = Number(positions[triangle.b + 2]);
  const cx = Number(positions[triangle.c]);
  const cy = Number(positions[triangle.c + 1]);
  const cz = Number(positions[triangle.c + 2]);
  const denominator = (bx - ax) * (cz - az) - (cx - ax) * (bz - az);
  if (Math.abs(denominator) <= 1e-9) return;
  const spacing = cellSize / COVERAGE_RESOLUTION;
  const minimumGridX = Math.floor(Math.min(ax, bx, cx) / spacing);
  const maximumGridX = Math.floor(Math.max(ax, bx, cx) / spacing);
  const minimumGridZ = Math.floor(Math.min(az, bz, cz) / spacing);
  const maximumGridZ = Math.floor(Math.max(az, bz, cz) / spacing);
  for (let gridZ = minimumGridZ; gridZ <= maximumGridZ; gridZ++) {
    const sampleZ = (gridZ + 0.5) * spacing;
    for (let gridX = minimumGridX; gridX <= maximumGridX; gridX++) {
      const sampleX = (gridX + 0.5) * spacing;
      const deltaX = sampleX - ax;
      const deltaZ = sampleZ - az;
      const baryB = (deltaX * (cz - az) - (cx - ax) * deltaZ) / denominator;
      const baryC = ((bx - ax) * deltaZ - deltaX * (bz - az)) / denominator;
      const baryA = 1 - baryB - baryC;
      if (baryA < -1e-7 || baryB < -1e-7 || baryC < -1e-7) continue;
      const cellX = Math.floor(gridX / COVERAGE_RESOLUTION);
      const cellZ = Math.floor(gridZ / COVERAGE_RESOLUTION);
      const localX = ((gridX % COVERAGE_RESOLUTION) + COVERAGE_RESOLUTION) % COVERAGE_RESOLUTION;
      const localZ = ((gridZ % COVERAGE_RESOLUTION) + COVERAGE_RESOLUTION) % COVERAGE_RESOLUTION;
      visit(
        cellX,
        cellZ,
        localZ * COVERAGE_RESOLUTION + localX,
        ay * baryA + by * baryB + cy * baryC
      );
    }
  }
}

export function eligibleTriangles(
  primitive: ShadoWorldPrimitive,
  minimumUpNormal: number
): SurfaceTriangle[] {
  const output: SurfaceTriangle[] = [];
  const positions = primitive.positions;
  let totalArea = 0;
  for (let index = 0; index + 2 < primitive.indices.length; index += 3) {
    const a = Number(primitive.indices[index]) * 3;
    const b = Number(primitive.indices[index + 1]) * 3;
    const c = Number(primitive.indices[index + 2]) * 3;
    const abX = Number(positions[b]) - Number(positions[a]);
    const abY = Number(positions[b + 1]) - Number(positions[a + 1]);
    const abZ = Number(positions[b + 2]) - Number(positions[a + 2]);
    const acX = Number(positions[c]) - Number(positions[a]);
    const acY = Number(positions[c + 1]) - Number(positions[a + 1]);
    const acZ = Number(positions[c + 2]) - Number(positions[a + 2]);
    const normalX = abY * acZ - abZ * acY;
    const normalY = abZ * acX - abX * acZ;
    const normalZ = abX * acY - abY * acX;
    const doubleArea = Math.hypot(normalX, normalY, normalZ);
    if (doubleArea <= 1e-6 || Math.abs(normalY) / doubleArea < minimumUpNormal) {
      continue;
    }
    totalArea += doubleArea * 0.5;
    output.push({ a, b, c, cumulativeArea: totalArea });
  }
  return output;
}

export function appendSurfaceHeightField(
  output: NonNullable<NonNullable<ShadoWorldGrassPackage['coverage']>['heightField']>,
  coverageHeights: Float64Array
): void {
  const blockSize = COVERAGE_RESOLUTION / SURFACE_HEIGHT_RESOLUTION;
  const samples: number[] = [];
  let minimumY = Number.POSITIVE_INFINITY;
  let maximumY = Number.NEGATIVE_INFINITY;
  for (let row = 0; row < SURFACE_HEIGHT_RESOLUTION; row++) {
    for (let column = 0; column < SURFACE_HEIGHT_RESOLUTION; column++) {
      let totalY = 0;
      let count = 0;
      for (let dz = 0; dz < blockSize; dz++) {
        for (let dx = 0; dx < blockSize; dx++) {
          const x = column * blockSize + dx;
          const z = row * blockSize + dz;
          const value = coverageHeights[z * COVERAGE_RESOLUTION + x]!;
          if (!Number.isFinite(value)) continue;
          totalY += value;
          count++;
        }
      }
      const value = count ? totalY / count : Number.NaN;
      samples.push(value);
      if (Number.isFinite(value)) {
        minimumY = Math.min(minimumY, value);
        maximumY = Math.max(maximumY, value);
      }
    }
  }
  if (!Number.isFinite(minimumY)) {
    minimumY = 0;
    maximumY = 0;
  }
  const heightRange = maximumY - minimumY;
  output.minimumY.push(minimumY);
  output.heightRange.push(heightRange);
  for (let word = 0; word < output.wordsPerCell; word++) {
    let bits = 0;
    for (let bit = 0; bit < 32; bit++) {
      const sample = word * 32 + bit;
      if (sample < samples.length && Number.isFinite(samples[sample])) {
        bits = (bits | (1 << bit)) >>> 0;
      }
    }
    output.words.push(bits);
  }
  for (const value of samples) {
    output.samples.push(
      Number.isFinite(value) && heightRange > 1e-9
        ? Math.round(((value - minimumY) / heightRange) * SURFACE_HEIGHT_QUANTIZATION_MAX)
        : 0
    );
  }
}

export function randomGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return state / 0x1_0000_0000;
  };
}
