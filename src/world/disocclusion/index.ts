/**
 * Runtime-safe surface of the disocclusion PVS prototype. The baker lives in
 * './bake-entry' so a runtime bundle never pulls WGSL or the orchestration.
 */
export * from './types';
export * from './layers';
export * from './fixtures';
export * from './sidecar';
export * from './admission';
export * from './geometry';
export { propagateReference, countTiles, describeLayer, EMPTY_SAMPLE, TILE_EMPTY, TILE_OPEN, TILE_CLOSED } from './reference';
export { classifyBox, classifyTargets, targetBounds, volumetricFilter, rawVisibleTriangles } from './classify';
export type { DisocclusionTargetClassification, DisocclusionTargetReason } from './classify';
