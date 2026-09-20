import {
  MAX_REGIONS_PER_ENTITY,
  anyRegionAdmits,
  regionsForBounds,
} from '../src/world';
import type { RegionGrid } from '../src/world';

/** Four by four regions of 64 units, origin at the world corner. */
const grid: RegionGrid = { originX: -128, originZ: -128, size: 64, width: 4, height: 4 };
const scratch = new Uint32Array(MAX_REGIONS_PER_ENTITY);

const at = (x: number, z: number) => z * grid.width + x;

describe('region membership', () => {
  it('puts a small thing in exactly the region that holds it', () => {
    const { regions, overflow } = regionsForBounds(grid, -100, -100, -90, -90, scratch);
    expect(Array.from(regions)).toEqual([at(0, 0)]);
    expect(overflow).toBe(false);
  });

  it('puts a thing straddling a boundary in both regions', () => {
    // Crosses x = -64, the boundary between column 0 and column 1.
    const { regions } = regionsForBounds(grid, -70, -100, -58, -90, scratch);
    expect(Array.from(regions)).toEqual([at(0, 0), at(1, 0)]);
  });

  it('counts a bound that ends exactly on an edge as touching both', () => {
    /*
     * Stamps sit on round numbers and regions are powers of two, so exact
     * edge contact is common rather than exotic. The epsilon widens the box,
     * which is the safe direction.
     */
    const { regions } = regionsForBounds(grid, -64, -100, -64, -90, scratch);
    expect(Array.from(regions)).toEqual([at(0, 0), at(1, 0)]);
  });

  it('covers every region under a large footprint', () => {
    const { regions, overflow } = regionsForBounds(grid, -100, -100, 40, 40, scratch);
    expect(regions.length).toBe(9);
    expect(overflow).toBe(false);
  });

  it('clips at the grid edge without calling it unknown', () => {
    /*
     * There are no rows outside the grid and no camera either, so the regions
     * that exist describe everything that can see this entity. Calling the
     * overhang unknown would make every entity along a zone's border
     * permanently visible.
     */
    const { regions, overflow } = regionsForBounds(grid, -200, -100, -90, -90, scratch);
    expect(Array.from(regions)).toEqual([at(0, 0)]);
    expect(overflow).toBe(false);
  });

  it('returns nothing, without overflow, for a box wholly off the grid', () => {
    const { regions, overflow } = regionsForBounds(grid, 500, 500, 600, 600, scratch);
    expect(regions.length).toBe(0);
    expect(overflow).toBe(false);
  });

  it('overflows rather than truncating when the footprint exceeds the cap', () => {
    const wide: RegionGrid = { originX: 0, originZ: 0, size: 1, width: 200, height: 200 };
    const { regions, overflow } = regionsForBounds(wide, 0, 0, 20, 20, scratch);
    expect(regions.length).toBe(0);
    expect(overflow).toBe(true);
  });

  it('refuses to judge a nonfinite bound', () => {
    const { regions, overflow } = regionsForBounds(grid, NaN, 0, 10, 10, scratch);
    expect(regions.length).toBe(0);
    expect(overflow).toBe(true);
  });
});

describe('admission across regions', () => {
  const PVS = 0x01;
  const LOADED = 0x10;

  it('admits when one region satisfies every required bit', () => {
    const flags = [PVS | LOADED, 0, 0, 0];
    expect(anyRegionAdmits([0, 1], flags, PVS | LOADED)).toBe(true);
  });

  it('never combines bits from different regions into a pass', () => {
    /*
     * One region is in the PVS but not loaded; the other is loaded but not in
     * the PVS. Neither is a place this entity can be seen from, and or-ing
     * them would invent one.
     */
    const flags = [PVS, LOADED, 0, 0];
    expect(anyRegionAdmits([0, 1], flags, PVS | LOADED)).toBe(false);
  });

  it('ignores regions outside the flag array rather than reading past it', () => {
    expect(anyRegionAdmits([99], [PVS], PVS)).toBe(false);
  });
});
