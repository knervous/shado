import type { ShadoWorldCompiledPointLight } from './types';

/**
 * How a compiled point light behaves at runtime: when it is lit, and how it
 * flickers.
 *
 * This lives apart from `ShadoWorldLightBuffer` so that anything else which
 * must agree with the light about "is it lit right now" -- a flame particle on
 * a torch, a glowing pane on a lamp -- asks the same function the light field
 * asks, instead of re-deriving the rule and drifting from it. It deliberately
 * imports nothing at runtime (types only), so plain Node tests can load it.
 */
export type ShadoRuntimeLightBehavior = {
  mode: 'always' | 'night' | 'schedule';
  onHour: number;
  offHour: number;
  transitionMinutes: number;
  flicker: 'steady' | 'flame' | 'wisp';
  amplitude: number;
  speed: number;
  seed: number;
};

/** The subset of a compiled light the behaviour is resolved from. */
export type ShadoRuntimeLightBehaviorSource = Pick<
  ShadoWorldCompiledPointLight,
  'id' | 'activation' | 'flicker' | 'metadata'
>;

/**
 * True when legacy light metadata names a flame: `fire|flame|torch|lantern|brazier`
 * in `metadata.kind` or `metadata.flickerProfile`. Flame-like lights default to
 * the dusk-to-dawn `night` schedule and the `flame` flicker.
 */
export function isShadoFlameLikeLightMetadata(metadata: Record<string, unknown> | undefined): boolean {
  const kind = typeof metadata?.kind === 'string' ? metadata.kind.toLowerCase() : '';
  const profile = typeof metadata?.flickerProfile === 'string' ? metadata.flickerProfile.toLowerCase() : '';
  return /fire|flame|torch|lantern|brazier/.test(`${kind} ${profile}`);
}

export function resolveShadoRuntimeLightBehavior(light: ShadoRuntimeLightBehaviorSource): ShadoRuntimeLightBehavior {
  const metadata = light.metadata ?? {};
  const legacyProfile = typeof metadata.flickerProfile === 'string'
    ? metadata.flickerProfile.toLowerCase()
    : '';
  const flameLike = isShadoFlameLikeLightMetadata(metadata);
  const legacyInterior = metadata.interior === true;
  const legacyMode = metadata.activationMode;
  const mode = light.activation?.mode ?? (
    legacyMode === 'always' || legacyMode === 'night' || legacyMode === 'schedule'
      ? legacyMode
      : legacyInterior ? 'always' : flameLike ? 'night' : 'always'
  );
  const profile = light.flicker?.profile ?? (
    legacyProfile.includes('wisp') ? 'wisp' : flameLike ? 'flame' : 'steady'
  );
  const legacyAmplitude = finiteMetadataNumber(metadata.flickerAmplitude);
  const legacySpeed = finiteMetadataNumber(metadata.flickerSpeed);
  return {
    mode,
    onHour: light.activation?.onHour ?? finiteMetadataNumber(metadata.onHour) ?? 18,
    offHour: light.activation?.offHour ?? finiteMetadataNumber(metadata.offHour) ?? 6,
    transitionMinutes:
      light.activation?.transitionMinutes ??
      finiteMetadataNumber(metadata.transitionMinutes) ??
      25,
    flicker: profile,
    amplitude: Math.min(0.5, Math.max(0,
      light.flicker?.amplitude ?? legacyAmplitude ?? (profile === 'steady' ? 0 : 0.065)
    )),
    speed: Math.min(30, Math.max(0,
      light.flicker?.speed ?? legacySpeed ?? (profile === 'wisp' ? 2.4 : 6.5)
    )),
    seed: hashLightId(light.id),
  };
}

/**
 * 0..1 operating level at a world-clock hour: 1 inside the schedule, 0 outside,
 * smoothstepped over `transitionMinutes` at both ends. `night` is fixed 18:00-06:00.
 */
export function shadoRuntimeLightScheduleLevel(behavior: ShadoRuntimeLightBehavior, hour: number): number {
  if (behavior.mode === 'always') return 1;
  const onHour = behavior.mode === 'night' ? 18 : wrapShadoHour(behavior.onHour);
  const offHour = behavior.mode === 'night' ? 6 : wrapShadoHour(behavior.offHour);
  const duration = positiveModulo(offHour - onHour, 24) || 24;
  const elapsed = positiveModulo(wrapShadoHour(hour) - onHour, 24);
  if (elapsed > duration) return 0;
  const fade = Math.min(duration * 0.5, Math.max(0, behavior.transitionMinutes / 60));
  if (fade <= 0) return 1;
  return Math.min(smoothstep(0, fade, elapsed), smoothstep(0, fade, duration - elapsed));
}

export function wrapShadoHour(hour: number): number {
  return positiveModulo(Number.isFinite(hour) ? hour : 12, 24);
}

function smoothstep(edge0: number, edge1: number, value: number): number {
  const t = Math.max(0, Math.min(1, (value - edge0) / Math.max(0.0001, edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function finiteMetadataNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function hashLightId(id: string): number {
  let hash = 2166136261;
  for (let index = 0; index < id.length; index++) {
    hash ^= id.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 0xffffffff;
}
