/**
 * Vegetation predicates, shared by the bake and the client so the two can
 * never disagree about what a prototype is.
 *
 * There are two distinct questions, and they split the vocabulary:
 *
 * - "May grass grow beneath it?" — true for ALL vegetation, trees included. A
 *   canopy's upward-facing leaves must not clear a circle of lawn.
 * - "Does it render as a transient instance?" — true only for understory.
 *   Trees are explicitly placed landmarks and stay on the stamped-object
 *   layer with its irradiance and durable identity; shrubs, flowers, ferns
 *   and the like are ground-cover decoration drawn through the foliage
 *   container with no physics.
 */

const UNDERSTORY = /(?:^|-)(?:shrub|bush|fern|bracken|gorse|ivy|flower)-/;
const CANOPY = /(?:^|-)(?:foliage|tree|sapling)-/;

/**
 * Vegetation for grass purposes: grass keeps growing beneath it.
 *
 * Authored prototypes declare themselves with `semanticRole: 'foliage'`. The id
 * vocabulary covers promoted generator output, which predates that field and
 * ships an empty metadata object. It deliberately excludes container words like
 * `planter`, which are usually stone and should suppress the grass under them.
 */
export function isShadoWorldFoliageMetadata(
  id: string,
  metadata: Readonly<Record<string, unknown>> = {}
): boolean {
  if (metadata.semanticRole === 'foliage') return true;
  // Explicit collision means the prototype is walked on or bumped into, which
  // makes it architecture whatever it is called.
  if (metadata.collisionPolicy === 'explicit') return false;
  return UNDERSTORY.test(id) || CANOPY.test(id);
}

/**
 * Understory only: the prototypes the client renders as transient foliage
 * instances instead of durable stamped objects.
 */
export function isShadoWorldTransientFoliageMetadata(
  id: string,
  metadata: Readonly<Record<string, unknown>> = {}
): boolean {
  if (metadata.collisionPolicy === 'explicit') return false;
  // The authored override wins outright, so Libra can force either rendering
  // path per prototype without renaming it.
  if (metadata.transientFoliage === false) return false;
  if (metadata.transientFoliage === true) return true;
  // Understory words win over the generic canopy words, so a
  // 'tcw-foliage-fern-v1' is a fern, not a tree that happens to say foliage.
  if (UNDERSTORY.test(id)) return true;
  return metadata.semanticRole === 'foliage' && !CANOPY.test(id);
}
