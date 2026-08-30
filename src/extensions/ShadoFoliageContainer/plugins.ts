import { BABYLON } from '../../babylon';
import type { ShadoMaterial } from '../../materials/ShadoMaterial';

/**
 * Per-frame state every foliage plugin may read.
 *
 * `focus` is normally the player, not the camera. Foliage should react to who
 * is walking through it even while the camera orbits.
 */
export type ShadoFoliageFrame = {
  timeSeconds: number;
  focus: readonly [number, number, number];
};

/**
 * Shader source a plugin contributes.
 *
 * `displace` is the only non-standard slot. It runs inside the foliage
 * displacement block that the container emits, where these locals are in scope
 * and are the plugin's entire mutable surface:
 *
 * - `shadoFoliageWorld` — mutable world position of this vertex.
 * - `shadoFoliageRoot`  — world position of the instance origin.
 * - `shadoFoliageAnchor` — the individual plant's own root. Distance-based
 *   behavior (gust travel, fades, player bend) must measure from this, never
 *   from `shadoFoliageRoot`: for derived plants such as grass blades the
 *   instance origin is the cell corner, and measuring from it makes every
 *   distance effect snap at cell granularity — square seams.
 * - `shadoFoliageUp`    — 0 at the root, 1 at the top of the source mesh.
 * - `shadoFoliageScale` — the instance's uniform scale.
 * - `shadoFoliageFade`  — mutable 0..1 coverage, forwarded to the fragment stage.
 * - `shadoFoliagePhase`, `shadoFoliageStiffness`, `shadoFoliageVariation` —
 *   mutable per-instance character, seeded from `inst.foliageParams`. A plugin
 *   that derives many plants from one instance overwrites these per plant.
 * - `shadoColor`        — mutable instance color, written to `vColor` afterwards.
 * - `inst`              — the actor header, including `foliageParams`.
 *
 * Plugins never write the clip position. The container writes it once, after
 * every plugin has run, so displacements compose instead of fighting.
 */
export type ShadoFoliageShaderFragments = {
  vertexDeclarations?: string;
  fragmentDeclarations?: string;
  vertexInstance?: string;
  displace?: string;
  fragmentSurface?: string;
};

export type ShadoFoliagePlugin = {
  readonly name: string;
  /** Uniform names this plugin owns. Declared to Babylon at material construction. */
  readonly uniforms: readonly string[];
  readonly glsl: ShadoFoliageShaderFragments;
  readonly wgsl: ShadoFoliageShaderFragments;
  bind(material: ShadoMaterial<any>, frame: ShadoFoliageFrame): void;
};

export type ShadoFoliageWindConfig = {
  /** World-space sway in metres at the top of the mesh, before instance scale. */
  amplitude?: number;
  /** Sway cycles per second. */
  frequency?: number;
  /** Additional amplitude that travels across the world as coherent gusts. */
  gustAmplitude?: number;
  gustFrequency?: number;
  /** Gust wavelength in metres. Larger values make broader, slower waves. */
  gustWavelength?: number;
  /** Horizontal wind direction. Normalized on construction. */
  direction?: readonly [number, number];
  /**
   * How strongly `foliageParams.y` (per-instance stiffness) resists the wind.
   * At 0 every instance sways identically; at 1 a stiff instance barely moves.
   */
  stiffnessInfluence?: number;
};

export type ShadoFoliageProximityFadeConfig = {
  /** Distance from the focus at which fading begins. */
  fadeStart: number;
  /** Distance at which the instance is fully gone. Must exceed `fadeStart`. */
  fadeEnd: number;
  /**
   * `dither` keeps the material opaque and drops fragments on a stable screen
   * pattern. `shrink` collapses the instance toward its root instead, which is
   * cheaper and reads better for grass than for a tree.
   */
  mode?: 'dither' | 'shrink';
};

export type ShadoFoliagePlayerBendConfig = {
  /** Radius in metres within which the focus pushes foliage aside. */
  radius: number;
  /** Peak horizontal displacement in metres at the top of the mesh. */
  strength: number;
};

export type ShadoFoliageTintConfig = {
  /** Color reached at `foliageParams.z == 1`. */
  variationColor: readonly [number, number, number];
  /** Blend weight toward `variationColor`. */
  variationStrength?: number;
  /** Darkening applied at the root, easing to none at the top. Fakes contact AO. */
  rootDarkening?: number;
};

/** Declarative plugin selection, so a configuration can travel as authored JSON. */
export type ShadoFoliagePluginSpec =
  | ({ plugin: 'wind' } & ShadoFoliageWindConfig)
  | ({ plugin: 'proximityFade' } & ShadoFoliageProximityFadeConfig)
  | ({ plugin: 'playerBend' } & ShadoFoliagePlayerBendConfig)
  | ({ plugin: 'tint' } & ShadoFoliageTintConfig);

const scratch = () => new BABYLON.Vector4(0, 0, 0, 0);

export function shadoFoliageWind(config: ShadoFoliageWindConfig = {}): ShadoFoliagePlugin {
  const amplitude = positive(config.amplitude ?? 0.18, 'wind.amplitude');
  const frequency = positive(config.frequency ?? 1.35, 'wind.frequency');
  const gustAmplitude = nonNegative(config.gustAmplitude ?? 0.32, 'wind.gustAmplitude');
  const gustFrequency = positive(config.gustFrequency ?? 0.22, 'wind.gustFrequency');
  const gustWavelength = positive(config.gustWavelength ?? 48, 'wind.gustWavelength');
  const stiffnessInfluence = unit(config.stiffnessInfluence ?? 0.6, 'wind.stiffnessInfluence');
  const [rawX, rawZ] = config.direction ?? [1, 0.35];
  const length = Math.hypot(rawX, rawZ);
  if (!(length > 1e-6)) throw new Error('foliage wind.direction must not be zero-length');
  const directionX = rawX / length;
  const directionZ = rawZ / length;

  const wind = scratch();
  const gust = scratch();

  return {
    name: 'wind',
    uniforms: ['uShadoFoliageWind', 'uShadoFoliageGust'],
    glsl: {
      vertexDeclarations: `
uniform vec4 uShadoFoliageWind;
uniform vec4 uShadoFoliageGust;`,
      displace: `
  {
    vec2 windDirection = uShadoFoliageWind.xy;
    float windPhase = shadoFoliagePhase * 6.2831853;
    float windSway = sin(uShadoFoliageTime * uShadoFoliageWind.w + windPhase);
    float gustTravel = dot(shadoFoliageAnchor.xz, windDirection) / uShadoFoliageGust.z;
    float windGust = sin(uShadoFoliageTime * uShadoFoliageGust.y - gustTravel) * 0.5 + 0.5;
    float windResistance = 1.0 - shadoFoliageStiffness * uShadoFoliageGust.w;
    float windBend = shadoFoliageUp * shadoFoliageUp * shadoFoliageScale * windResistance
      * (windSway * uShadoFoliageWind.z + windGust * uShadoFoliageGust.x);
    shadoFoliageWorld.x += windDirection.x * windBend;
    shadoFoliageWorld.z += windDirection.y * windBend;
  }`,
    },
    wgsl: {
      vertexDeclarations: `
uniform uShadoFoliageWind: vec4f;
uniform uShadoFoliageGust: vec4f;`,
      displace: `
  {
    let windDirection = uniforms.uShadoFoliageWind.xy;
    let windPhase = shadoFoliagePhase * 6.2831853;
    let windSway = sin(uniforms.uShadoFoliageTime * uniforms.uShadoFoliageWind.w + windPhase);
    let gustTravel = dot(shadoFoliageAnchor.xz, windDirection) / uniforms.uShadoFoliageGust.z;
    let windGust = sin(uniforms.uShadoFoliageTime * uniforms.uShadoFoliageGust.y - gustTravel) * 0.5 + 0.5;
    let windResistance = 1.0 - shadoFoliageStiffness * uniforms.uShadoFoliageGust.w;
    let windBend = shadoFoliageUp * shadoFoliageUp * shadoFoliageScale * windResistance
      * (windSway * uniforms.uShadoFoliageWind.z + windGust * uniforms.uShadoFoliageGust.x);
    shadoFoliageWorld = shadoFoliageWorld + vec3f(
      windDirection.x * windBend,
      0.0,
      windDirection.y * windBend
    );
  }`,
    },
    bind(material) {
      wind.set(directionX, directionZ, amplitude, frequency * 6.2831853);
      gust.set(gustAmplitude, gustFrequency * 6.2831853, gustWavelength, stiffnessInfluence);
      material.setVector4('uShadoFoliageWind', wind);
      material.setVector4('uShadoFoliageGust', gust);
    },
  };
}

export function shadoFoliageProximityFade(
  config: ShadoFoliageProximityFadeConfig
): ShadoFoliagePlugin {
  const fadeStart = positive(config.fadeStart, 'proximityFade.fadeStart');
  const fadeEnd = positive(config.fadeEnd, 'proximityFade.fadeEnd');
  if (fadeEnd <= fadeStart) {
    throw new Error('foliage proximityFade.fadeEnd must exceed proximityFade.fadeStart');
  }
  const mode = config.mode ?? 'dither';
  const fade = scratch();

  const glslShrink =
    mode === 'shrink'
      ? '\n    shadoFoliageWorld = mix(shadoFoliageAnchor, shadoFoliageWorld, fadeWeight);'
      : '';
  const wgslShrink =
    mode === 'shrink'
      ? '\n    shadoFoliageWorld = mix(shadoFoliageAnchor, shadoFoliageWorld, vec3f(fadeWeight));'
      : '';

  // Ordered dithering keeps the material opaque, so foliage never enters the
  // transparent sort. The threshold comes from the screen position, so a fading
  // instance stipples away in place instead of shimmering as the camera moves.
  const glslFragment =
    mode === 'dither'
      ? `
  {
    vec2 ditherCoord = floor(mod(gl_FragCoord.xy, 4.0));
    float ditherIndex = ditherCoord.y * 4.0 + ditherCoord.x;
    float ditherThreshold = (ditherIndex + 0.5) * 0.0625;
    if (vShadoFoliageFade < ditherThreshold) discard;
  }`
      : '';
  const wgslFragment =
    mode === 'dither'
      ? `
  {
    let ditherCoord = floor(fragmentInputs.position.xy % vec2f(4.0));
    let ditherIndex = ditherCoord.y * 4.0 + ditherCoord.x;
    let ditherThreshold = (ditherIndex + 0.5) * 0.0625;
    if (fragmentInputs.vShadoFoliageFade < ditherThreshold) { discard; }
  }`
      : '';

  return {
    name: 'proximityFade',
    uniforms: ['uShadoFoliageFade'],
    glsl: {
      vertexDeclarations: '\nuniform vec4 uShadoFoliageFade;',
      displace: `
  {
    float fadeDistance = length(shadoFoliageAnchor.xz - uShadoFoliageFocus.xz);
    float fadeWeight = 1.0 - smoothstep(uShadoFoliageFade.x, uShadoFoliageFade.y, fadeDistance);
    shadoFoliageFade *= fadeWeight;${glslShrink}
  }`,
      fragmentSurface: glslFragment,
    },
    wgsl: {
      vertexDeclarations: '\nuniform uShadoFoliageFade: vec4f;',
      displace: `
  {
    let fadeDistance = length(shadoFoliageAnchor.xz - uniforms.uShadoFoliageFocus.xz);
    let fadeWeight = 1.0 - smoothstep(
      uniforms.uShadoFoliageFade.x,
      uniforms.uShadoFoliageFade.y,
      fadeDistance
    );
    shadoFoliageFade = shadoFoliageFade * fadeWeight;${wgslShrink}
  }`,
      fragmentSurface: wgslFragment,
    },
    bind(material) {
      fade.set(fadeStart, fadeEnd, mode === 'dither' ? 1 : 0, 0);
      material.setVector4('uShadoFoliageFade', fade);
    },
  };
}

export function shadoFoliagePlayerBend(
  config: ShadoFoliagePlayerBendConfig
): ShadoFoliagePlugin {
  const radius = positive(config.radius, 'playerBend.radius');
  const strength = positive(config.strength, 'playerBend.strength');
  const bend = scratch();

  return {
    name: 'playerBend',
    uniforms: ['uShadoFoliageBend'],
    glsl: {
      vertexDeclarations: '\nuniform vec4 uShadoFoliageBend;',
      displace: `
  {
    vec2 bendOffset = shadoFoliageAnchor.xz - uShadoFoliageFocus.xz;
    float bendDistance = length(bendOffset);
    float bendFalloff = 1.0 - smoothstep(0.0, uShadoFoliageBend.x, bendDistance);
    vec2 bendDirection = bendDistance > 0.0001 ? bendOffset / bendDistance : vec2(1.0, 0.0);
    float bendAmount = bendFalloff * uShadoFoliageBend.y * shadoFoliageUp * shadoFoliageScale;
    shadoFoliageWorld.x += bendDirection.x * bendAmount;
    shadoFoliageWorld.z += bendDirection.y * bendAmount;
    // Bending is a rotation, not a stretch: give back the height the tip lost
    // travelling along its arc, so a pushed blade does not also grow taller.
    shadoFoliageWorld.y -= bendAmount * bendAmount * 0.5 / max(shadoFoliageScale, 0.0001);
  }`,
    },
    wgsl: {
      vertexDeclarations: '\nuniform uShadoFoliageBend: vec4f;',
      displace: `
  {
    let bendOffset = shadoFoliageAnchor.xz - uniforms.uShadoFoliageFocus.xz;
    let bendDistance = length(bendOffset);
    let bendFalloff = 1.0 - smoothstep(0.0, uniforms.uShadoFoliageBend.x, bendDistance);
    let bendDirection = select(
      vec2f(1.0, 0.0),
      bendOffset / max(bendDistance, 0.0001),
      bendDistance > 0.0001
    );
    let bendAmount = bendFalloff * uniforms.uShadoFoliageBend.y * shadoFoliageUp * shadoFoliageScale;
    shadoFoliageWorld = shadoFoliageWorld + vec3f(
      bendDirection.x * bendAmount,
      -bendAmount * bendAmount * 0.5 / max(shadoFoliageScale, 0.0001),
      bendDirection.y * bendAmount
    );
  }`,
    },
    bind(material) {
      bend.set(radius, strength, 0, 0);
      material.setVector4('uShadoFoliageBend', bend);
    },
  };
}

export function shadoFoliageTint(config: ShadoFoliageTintConfig): ShadoFoliagePlugin {
  const [red, green, blue] = config.variationColor;
  unit(red, 'tint.variationColor.r');
  unit(green, 'tint.variationColor.g');
  unit(blue, 'tint.variationColor.b');
  const variationStrength = unit(config.variationStrength ?? 1, 'tint.variationStrength');
  const rootDarkening = unit(config.rootDarkening ?? 0.45, 'tint.rootDarkening');
  const tint = scratch();
  const weights = scratch();

  return {
    name: 'tint',
    uniforms: ['uShadoFoliageTint', 'uShadoFoliageTintWeights'],
    glsl: {
      vertexDeclarations: `
uniform vec4 uShadoFoliageTint;
uniform vec4 uShadoFoliageTintWeights;`,
      displace: `
  {
    float tintWeight = shadoFoliageVariation * uShadoFoliageTintWeights.x;
    float rootShade = mix(1.0 - uShadoFoliageTintWeights.y, 1.0, shadoFoliageUp);
    shadoColor.rgb = mix(shadoColor.rgb, uShadoFoliageTint.rgb, tintWeight) * rootShade;
  }`,
    },
    wgsl: {
      vertexDeclarations: `
uniform uShadoFoliageTint: vec4f;
uniform uShadoFoliageTintWeights: vec4f;`,
      displace: `
  {
    let tintWeight = shadoFoliageVariation * uniforms.uShadoFoliageTintWeights.x;
    let rootShade = mix(1.0 - uniforms.uShadoFoliageTintWeights.y, 1.0, shadoFoliageUp);
    shadoColor = vec4f(
      mix(shadoColor.rgb, uniforms.uShadoFoliageTint.rgb, vec3f(tintWeight)) * rootShade,
      shadoColor.a
    );
  }`,
    },
    bind(material) {
      tint.set(red, green, blue, 1);
      weights.set(variationStrength, rootDarkening, 0, 0);
      material.setVector4('uShadoFoliageTint', tint);
      material.setVector4('uShadoFoliageTintWeights', weights);
    },
  };
}

const FACTORIES = {
  wind: shadoFoliageWind,
  proximityFade: shadoFoliageProximityFade,
  playerBend: shadoFoliagePlayerBend,
  tint: shadoFoliageTint,
} as const;

/** Builds the plugin list for a declarative, serializable configuration. */
export function resolveShadoFoliagePlugins(
  specs: readonly ShadoFoliagePluginSpec[]
): ShadoFoliagePlugin[] {
  const seen = new Set<string>();
  return specs.map(spec => {
    const factory = FACTORIES[spec.plugin as keyof typeof FACTORIES];
    if (!factory) throw new Error(`Unknown foliage plugin '${spec.plugin}'`);
    // Two instances of one plugin would silently overwrite each other's
    // uniforms, leaving only the last configuration's behavior.
    if (seen.has(spec.plugin)) {
      throw new Error(`Foliage plugin '${spec.plugin}' is configured more than once`);
    }
    seen.add(spec.plugin);
    return (factory as (config: unknown) => ShadoFoliagePlugin)(spec);
  });
}

function positive(value: number, field: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`foliage ${field} must be positive`);
  return value;
}

function nonNegative(value: number, field: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`foliage ${field} must not be negative`);
  }
  return value;
}

function unit(value: number, field: string): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`foliage ${field} must be between zero and one`);
  }
  return value;
}
