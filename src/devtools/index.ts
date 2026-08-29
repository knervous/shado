/**
 * Node-only shado development tools (`@knervous/shado/devtools`).
 *
 * Headless WebGPU through Dawn plus preview rendering, for iterating on models
 * and scenes without a browser and for proving a pipeline stage by stage —
 * raw GLB, bake output, finalized runtime artifact — from one camera.
 *
 * Kept out of `@knervous/shado` proper so no browser bundle can reach node:zlib
 * or the Dawn binding.
 */
export {
  useBabylonRuntime,
  hasHostBabylonRuntime,
  babylonImport,
  type BabylonModuleImporter,
} from './babylon';
export { installHeadlessWebGpu, createHeadlessCanvas, BUFFER_USAGE, TEXTURE_USAGE, type HeadlessGpu } from './headless-gpu';
export { encodePng } from './png';
export {
  installHeadlessBasisTranscoder,
  loadBasisFile,
  transcodeBasisLayers,
  type BasisImage,
} from './basis';
export {
  GlbRenderer,
  renderGlb,
  renderGlbAssembly,
  type GlbAssemblyEntry,
  type GlbAssemblyRenderResult,
  type GlbRenderOptions,
  type GlbRenderRequest,
  type GlbRenderResult,
  type GlbRenderedView,
} from './glb-render-compat';
export { installImageDecoder, type DecodedImage, type ImageDecoder } from './headless-gpu';
export { MULTIVIEWS, NAMED_VIEWS, RAISED_BETA, coverage, viewAngles } from './views';
export {
  createPreviewSession,
  type PreviewSession,
  type SessionOptions,
  type SceneOptions,
  type CaptureOptions,
} from './session';
export {
  createHeadlessPreview,
  differenceImage,
  type CameraFraming,
  type HeadlessPreview,
  type PreviewImage,
  type PreviewOptions,
  type GlbPlacement,
  type RenderedView,
  type ViewRenderOptions,
} from './preview';
