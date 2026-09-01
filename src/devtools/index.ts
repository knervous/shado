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
export { installHeadlessWebGpu, createHeadlessCanvas, DEFAULT_DAWN_MODULE, BUFFER_USAGE, TEXTURE_USAGE, type HeadlessGpu } from './headless-gpu';
export { encodePng } from './png';
export {
  createLitePreviewSession,
  type LiteCameraFraming,
  type LitePreviewSession,
  type LiteSceneOptions,
  type LiteSessionOptions,
} from './lite-session';
export { createYuvConverter, supportsGpuYuv, type YuvConverter, type YuvConverterOptions } from './yuv';
// Exported so a consumer can install the transcoder itself, or point the
// shared-store resolver at a checkout laid out differently.
export { installHeadlessKtx2Transcoder } from './ktx2';
export { resolveSharedTextureUri, setSharedTextureRoot } from './shared-textures';
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
  type LoadOptions,
  type PreviewSession,
  type RawFrame,
  type UriResolver,
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
