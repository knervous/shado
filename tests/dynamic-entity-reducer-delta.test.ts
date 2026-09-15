import { describe, expect, it } from '@jest/globals';

import {
  createShadoDynamicEntityReducer,
  SHADO_DYNAMIC_ENTITY_EXPIRATION_RECORD_BYTES,
  SHADO_DYNAMIC_ENTITY_VISIBLE,
  SHADO_ENTITY2D_REDUCER_LAYOUT,
  ShadoDynamicEntityReducerOp,
} from '../src/render/ShadoDynamicEntityReducers';

describe('ShadoDynamicEntityReducer.applyDelta', () => {
  it('reuses its delta buffer, so a long-running host does not grow wasm memory', async () => {
    const reducer = await createShadoDynamicEntityReducer();
    const capacity = 64;
    const layout = SHADO_ENTITY2D_REDUCER_LAYOUT;
    const arena = reducer.initArena({
      entityBasePtr: reducer.alloc(capacity * layout.strideBytes),
      entityCapacity: capacity,
      entityStrideBytes: layout.strideBytes,
      positionSizeOffset: layout.positionSizeOffset,
      renderOffset: layout.renderOffset,
      destinationSizeOffset: layout.destinationSizeOffset,
      motionOffset: layout.motionOffset,
      renderStateOffset: layout.renderStateOffset,
      activeIndexPtr: reducer.alloc(capacity * 4),
      activeIndexCapacity: capacity,
      changedIndexPtr: reducer.alloc(capacity * 4),
      changedIndexCapacity: capacity,
      expirationPtr: reducer.alloc(capacity * SHADO_DYNAMIC_ENTITY_EXPIRATION_RECORD_BYTES),
      expirationCapacity: capacity,
      expirationStrideBytes: SHADO_DYNAMIC_ENTITY_EXPIRATION_RECORD_BYTES,
    });
    const records = Array.from({ length: capacity }, (_, index) => ({
      op: ShadoDynamicEntityReducerOp.DirectPlace,
      index,
      x: index,
      y: 0,
      width: 1,
      depth: 1,
      flags: SHADO_DYNAMIC_ENTITY_VISIBLE,
    }));
    reducer.applyDelta(records);
    reducer.clearChanged();
    const memoryBytes = reducer.memory.buffer.byteLength;

    // ~2.7 MB of deltas in total: enough to grow memory if each call allocated.
    for (let round = 0; round < 1000; round++) {
      records[round % capacity].x = round;
      expect(reducer.applyDelta(records)).toBe(capacity);
      reducer.clearChanged();
    }

    expect(reducer.memory.buffer.byteLength).toBe(memoryBytes);
    const view = new DataView(reducer.memory.buffer);
    const lastIndex = 999 % capacity;
    expect(
      view.getFloat32(arena.entityBasePtr + lastIndex * layout.strideBytes + layout.positionSizeOffset, true)
    ).toBe(999);
  });
});
