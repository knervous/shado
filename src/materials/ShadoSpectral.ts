/**
 * Spectral shading: drawing a body as something that is only partly there.
 *
 * A hand-written fragment transform rather than a generated one, paired
 * GLSL/WGSL, in the same shape as the world light field beside it — an
 * application splices these chunks into its own shader and calls the entry
 * points. It lives in Shado because both halves must stay identical and because
 * nothing about it is specific to one game's material: given a world position,
 * a height up the body and a colour, it decides how solid this pixel is and
 * what is left of it.
 *
 * `shadoSpectralDensity` returns a coverage, and a host spends it one of two
 * ways:
 *
 *  - **As alpha**, if it can afford a second blended draw for its spectral
 *    instances. This is the good one, and it is what Eltania does: blending and
 *    depth writing are pipeline state rather than per-instance state, so the
 *    dead are submitted as their own draw over the same geometry, with blending
 *    on and depth writes off, while the living keep the solid pipeline they
 *    always had. Nothing is duplicated but a mesh and a matrix buffer.
 *  - **Against `shadoSpectralDither`**, if it cannot. A stochastic screen-door
 *    keeps a spectral body in the opaque pass with depth writes intact, which
 *    costs nothing to sort — but it is visibly grain, and at low coverage it is
 *    visibly *patterned* grain, which reads as a rendering fault rather than as
 *    a ghost. Reach for it only when a second draw is genuinely out of reach.
 *
 * WHAT MAKES IT READ AS A GHOST RATHER THAN AS A BROKEN MESH. The coverage
 * itself, however it is spent:
 *
 *  1. **It moves, in world space.** A static coverage makes a body look
 *     stencilled. A drifting three-dimensional noise field, stretched tall and
 *     narrow so its features are vertical streaks, makes the same body look
 *     like it is being carried off in wisps — and because the field is in world
 *     space rather than screen space, the wisps stay on the body as the camera
 *     moves and flow through it as it walks.
 *  2. **It thins from the hem upward.** Almost every convincing ghost is solid
 *     at the head and gone at the feet. This also removes the worst artefact of
 *     an even fade: thin geometry — fingers, boots, a nose — keeping just
 *     enough coverage to read as spiky, pointed fragments.
 *  3. **The same field shades what survives.** One sample decides both how
 *     solid a pixel is and how bright it is, so a spectral body has internal
 *     structure instead of reading as one flat wash.
 *
 * The silhouette is only *slightly* firmed up. Adding a strong edge term is the
 * obvious way to keep a fading body legible and it is what makes it look
 * pointy, because the term peaks exactly on the thin geometry that is already
 * over-represented. It is kept low and multiplied by presence so a faint body
 * does not get a hard outline it has not earned.
 */

/**
 * Uniforms a host shader must declare for the spectral chunks.
 *
 * Only the clock: everything else the transform needs arrives as arguments, so
 * a host can drive it from a varying, a uniform or a constant as it likes.
 */
export const SHADO_SPECTRAL_UNIFORMS = ['uShadoSpectralTime'] as const;

/**
 * The fragment transform, WebGPU.
 *
 * `shadoSpectralWisp` is sampled once and handed to both entry points: the same
 * field that decides how much of a pixel survives also shades what is left, so
 * the surviving pixels are bright exactly where the body is densest.
 */
export const SHADO_SPECTRAL_WGSL = /* wgsl */ `
/** Screen-door threshold, for a host with no blended draw to spend alpha in. */
fn shadoSpectralDither(fragment: vec2f) -> f32 {
  // Interleaved gradient noise: low-discrepancy, so a coverage lands as an even
  // scatter rather than as the 4x4 lattice an ordered Bayer matrix leaves. It
  // is still visibly a pattern at low coverage — see the header.
  return fract(52.9829189 * fract(dot(fragment, vec2f(0.06711056, 0.00583715))));
}

fn shadoSpectralHash(cell: vec3f) -> f32 {
  return fract(sin(dot(cell, vec3f(127.1, 311.7, 74.7))) * 43758.5453123);
}

fn shadoSpectralValue(p: vec3f) -> f32 {
  let base = floor(p);
  let t = smoothstep(vec3f(0.0), vec3f(1.0), p - base);
  let c000 = shadoSpectralHash(base + vec3f(0.0, 0.0, 0.0));
  let c100 = shadoSpectralHash(base + vec3f(1.0, 0.0, 0.0));
  let c010 = shadoSpectralHash(base + vec3f(0.0, 1.0, 0.0));
  let c110 = shadoSpectralHash(base + vec3f(1.0, 1.0, 0.0));
  let c001 = shadoSpectralHash(base + vec3f(0.0, 0.0, 1.0));
  let c101 = shadoSpectralHash(base + vec3f(1.0, 0.0, 1.0));
  let c011 = shadoSpectralHash(base + vec3f(0.0, 1.0, 1.0));
  let c111 = shadoSpectralHash(base + vec3f(1.0, 1.0, 1.0));
  let x00 = mix(c000, c100, t.x);
  let x10 = mix(c010, c110, t.x);
  let x01 = mix(c001, c101, t.x);
  let x11 = mix(c011, c111, t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}

fn shadoSpectralWisp(worldPosition: vec3f, time: f32) -> f32 {
  // Mostly upward, with a slow sideways lean so a standing body still moves.
  // Squashed in y, so the field's features stretch into vertical streaks: a
  // body shedding upward rather than one eaten by round holes.
  // Tall and narrow: high frequency across the body, low frequency up it, so
  // the field's features are vertical streaks rather than round holes. A body
  // is roughly two units across and six tall, so these numbers put two or three
  // streaks across a torso and barely one along its height.
  let axis = vec3f(1.8, 0.4, 1.8);
  let drift = vec3f(time * 0.05, -time * 0.5, time * 0.09);
  let coarse = shadoSpectralValue(worldPosition * axis * 0.6 + drift);
  let fine = shadoSpectralValue(worldPosition * axis * 2.0 + drift * 2.3);
  return clamp(coarse * 0.7 + fine * 0.3, 0.0, 1.0);
}

/** How solid this pixel is, 0..1: an alpha, or a screen-door threshold. */
fn shadoSpectralDensity(
  presence: f32,
  bodyHeight: f32,
  facing: f32,
  wisp: f32
) -> f32 {
  // Solid at the chest, gone at the hem.
  let rise = smoothstep(-0.05, 0.62, bodyHeight);
  let column = mix(0.16, 1.0, rise);
  let breath = mix(0.12, 1.95, wisp);
  // Deliberately small, and scaled by presence — see the header.
  let edge = facing * facing * 0.3;
  return clamp(presence * (column * breath + edge), 0.0, 1.0);
}

fn shadoSpectralSurface(
  lit: vec3f,
  albedoLuminance: f32,
  tint: vec3f,
  drain: f32,
  rim: f32,
  glow: f32,
  facing: f32,
  wisp: f32
) -> vec3f {
  // The drain is how far this body stops obeying the world's light and lights
  // itself: at 1 it is its own albedo's luminance and nothing else, so a
  // spectre in a sealed room is as visible as one in a lit hall.
  let litLuminance = dot(lit, vec3f(0.2126, 0.7152, 0.0722));
  // A clothed body's albedo luminance sits near 0.3, which lands almost black
  // once a cold tint multiplies it. Lift it before the tint, not after, so the
  // colour stays the authored one.
  var body = mix(litLuminance, albedoLuminance, clamp(drain, 0.0, 1.0)) * 1.45;
  // The same field that thins the body also shades it, so what survives has
  // internal structure rather than reading as one flat wash.
  body *= mix(0.62, 1.55, wisp);
  let halo = pow(facing, 2.6) * rim;
  return tint * (glow + body + halo);
}
`;

/** The fragment transform, WebGL2. Mirrors the WGSL above line for line. */
export const SHADO_SPECTRAL_GLSL = /* glsl */ `
/** Screen-door threshold — see the WGSL twin. */
float shadoSpectralDither(vec2 fragment) {
  return fract(52.9829189 * fract(dot(fragment, vec2(0.06711056, 0.00583715))));
}

float shadoSpectralHash(vec3 cell) {
  return fract(sin(dot(cell, vec3(127.1, 311.7, 74.7))) * 43758.5453123);
}

float shadoSpectralValue(vec3 p) {
  vec3 base = floor(p);
  vec3 t = smoothstep(vec3(0.0), vec3(1.0), p - base);
  float c000 = shadoSpectralHash(base + vec3(0.0, 0.0, 0.0));
  float c100 = shadoSpectralHash(base + vec3(1.0, 0.0, 0.0));
  float c010 = shadoSpectralHash(base + vec3(0.0, 1.0, 0.0));
  float c110 = shadoSpectralHash(base + vec3(1.0, 1.0, 0.0));
  float c001 = shadoSpectralHash(base + vec3(0.0, 0.0, 1.0));
  float c101 = shadoSpectralHash(base + vec3(1.0, 0.0, 1.0));
  float c011 = shadoSpectralHash(base + vec3(0.0, 1.0, 1.0));
  float c111 = shadoSpectralHash(base + vec3(1.0, 1.0, 1.0));
  float x00 = mix(c000, c100, t.x);
  float x10 = mix(c010, c110, t.x);
  float x01 = mix(c001, c101, t.x);
  float x11 = mix(c011, c111, t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}

float shadoSpectralWisp(vec3 worldPosition, float time) {
  vec3 axis = vec3(1.8, 0.4, 1.8);
  vec3 drift = vec3(time * 0.05, -time * 0.5, time * 0.09);
  float coarse = shadoSpectralValue(worldPosition * axis * 0.6 + drift);
  float fine = shadoSpectralValue(worldPosition * axis * 2.0 + drift * 2.3);
  return clamp(coarse * 0.7 + fine * 0.3, 0.0, 1.0);
}

float shadoSpectralDensity(
  float presence,
  float bodyHeight,
  float facing,
  float wisp
) {
  float rise = smoothstep(-0.05, 0.62, bodyHeight);
  float column = mix(0.16, 1.0, rise);
  float breath = mix(0.12, 1.95, wisp);
  float edge = facing * facing * 0.3;
  return clamp(presence * (column * breath + edge), 0.0, 1.0);
}

vec3 shadoSpectralSurface(
  vec3 lit,
  float albedoLuminance,
  vec3 tint,
  float drain,
  float rim,
  float glow,
  float facing,
  float wisp
) {
  float litLuminance = dot(lit, vec3(0.2126, 0.7152, 0.0722));
  float body = mix(litLuminance, albedoLuminance, clamp(drain, 0.0, 1.0)) * 1.45;
  body *= mix(0.62, 1.55, wisp);
  float halo = pow(facing, 2.6) * rim;
  return tint * (glow + body + halo);
}
`;
