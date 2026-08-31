import {
  isShadoWorldFoliageMetadata,
  isShadoWorldTransientFoliageMetadata,
} from '../src/world/foliage';
import { stampBlocksGrass } from '../src/preprocess/world-core';

const prototype = (id: string, metadata: Record<string, unknown> = {}) => ({ id, metadata });
const stamp = (metadata: Record<string, unknown> = {}) => ({ metadata });

describe('isShadoWorldFoliageMetadata', () => {
  it('recognizes authored and promoted foliage', () => {
    expect(isShadoWorldFoliageMetadata('anything', { semanticRole: 'foliage' })).toBe(true);
    expect(isShadoWorldFoliageMetadata('tcw-foliage-salt-oak-11m-v1')).toBe(true);
    expect(isShadoWorldFoliageMetadata('tcw-garden-shrub-v1')).toBe(true);
    expect(isShadoWorldFoliageMetadata('gm-bracken-clump-v1')).toBe(true);
    expect(isShadoWorldFoliageMetadata('gm-dead-tree-v1')).toBe(true);
    // A container, usually stone, so grass must not grow up through it.
    expect(isShadoWorldFoliageMetadata('tcw-stone-planter-v1')).toBe(false);
  });

  it('treats anything walked on as architecture whatever it is called', () => {
    expect(
      isShadoWorldFoliageMetadata('tcw-foliage-hedge-walkway-v1', {
        collisionPolicy: 'explicit',
      })
    ).toBe(false);
  });

  it('keeps grass under trees but renders them as placed objects, not transients', () => {
    for (const id of [
      'tcw-foliage-salt-oak-11m-v1',
      'tcw-courtyard-sapling-v1',
      'gm-dead-tree-v1',
    ]) {
      // Grass grows beneath a canopy...
      expect(isShadoWorldFoliageMetadata(id)).toBe(true);
      // ...but a tree is an explicitly placed landmark, never a transient.
      expect(isShadoWorldTransientFoliageMetadata(id)).toBe(false);
    }
    for (const id of ['tcw-garden-shrub-v1', 'gm-bracken-clump-v1']) {
      expect(isShadoWorldFoliageMetadata(id)).toBe(true);
      expect(isShadoWorldTransientFoliageMetadata(id)).toBe(true);
    }
    // Understory words win over the generic canopy vocabulary.
    expect(isShadoWorldTransientFoliageMetadata('tcw-foliage-fern-v1')).toBe(true);
    // An authored understory prototype needs no id vocabulary at all.
    expect(
      isShadoWorldTransientFoliageMetadata('tcw-flowerbed-a', { semanticRole: 'foliage' })
    ).toBe(true);
  });

  it('honours the explicit transientFoliage override in both directions', () => {
    // Libra sets this on the prototype so an artist can force either path.
    expect(
      isShadoWorldTransientFoliageMetadata('tcw-garden-shrub-v1', { transientFoliage: false })
    ).toBe(false);
    expect(
      isShadoWorldTransientFoliageMetadata('tcw-decorative-reed-cluster-v1', {
        transientFoliage: true,
      })
    ).toBe(true);
    // Walked-on things stay architecture even when someone flags them.
    expect(
      isShadoWorldTransientFoliageMetadata('tcw-hedge-walkway-v1', {
        transientFoliage: true,
        collisionPolicy: 'explicit',
      })
    ).toBe(false);
  });

  it('does not classify paving or structure as foliage', () => {
    for (const id of [
      'tcw-crownward-lane-cobble-4m-v1',
      'tcw-crownward-lane-junction-v1',
      'tcw-v26-road-east-plaza-link-tcw-civic-road-v26',
      'tcw-crownward-house-roof-bay-v1',
    ]) {
      expect(isShadoWorldFoliageMetadata(id)).toBe(false);
    }
  });
});

describe('stampBlocksGrass', () => {
  it('blocks by default, so an unflagged paving kit cannot grow grass through itself', () => {
    // The original defect: this prototype sets no metadata at all, and under
    // the previous opt-in rule contributed nothing to the coverage mask.
    expect(stampBlocksGrass(prototype('tcw-crownward-lane-cobble-4m-v1'), stamp())).toBe(true);
    expect(stampBlocksGrass(prototype('tcw-crownward-lane-drain-v1'), stamp())).toBe(true);
  });

  it('lets grass grow under foliage, which is the one thing it must not clear', () => {
    expect(stampBlocksGrass(prototype('tcw-foliage-salt-oak-11m-v1'), stamp())).toBe(false);
    expect(
      stampBlocksGrass(prototype('anything', { semanticRole: 'foliage' }), stamp())
    ).toBe(false);
  });

  it('honours an explicit flag in either direction', () => {
    expect(
      stampBlocksGrass(prototype('tcw-foliage-salt-oak-11m-v1', { grassBlocker: true }), stamp())
    ).toBe(true);
    expect(
      stampBlocksGrass(prototype('tcw-crownward-lane-cobble-4m-v1'), stamp({ grassBlocker: false }))
    ).toBe(false);
  });

  it('lets a stamp override its prototype', () => {
    const paving = prototype('tcw-crownward-lane-cobble-4m-v1', { grassBlocker: true });
    expect(stampBlocksGrass(paving, stamp({ grassBlocker: false }))).toBe(false);
    const tree = prototype('tcw-foliage-salt-oak-11m-v1');
    expect(stampBlocksGrass(tree, stamp({ grassBlocker: true }))).toBe(true);
  });
});
