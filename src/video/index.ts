/**
 * Video capture for shado — render, encode, transport (`@knervous/shado/video`).
 *
 * Three seams that do not know about each other: a {@link FrameSource} produces
 * frames, a {@link ShadoVideoEncoder} compresses them into a container, and a
 * {@link VideoSink} carries the bytes somewhere. Swapping the destination of a
 * capture from a file to a socket changes one argument.
 *
 * This entry point is runtime-neutral: it imports neither node nor the DOM, so
 * the same driver runs a headless Dawn render offline and a live canvas in the
 * browser. The ffmpeg encoder and the file sink are node-only and live in
 * `@knervous/shado/video/node`.
 *
 * Note what belongs to which seam, because the three usual names for this are
 * not peers: **WebCodecs is an encoder**, **MSE is a receiver**, and a
 * WebSocket or data channel is a **transport**. Only the last of those is a
 * sink.
 *
 * Offline, to a file:
 *
 *     const source = await createSessionFrameSource(session, {
 *       width: 1280, height: 720, camera,
 *       onFrame: orbitCamera(camera, { seconds: 6 }),
 *     });
 *     await renderVideo({ source, encoder, sink, seconds: 6, fps: 30 });
 *
 * A custom entrypoint, where a script is handed the scene and drives it —
 * animations, rigs, the shado runtime, a camera on a path:
 *
 *     const { source, seconds } = await createScriptFrameSource({
 *       script, session, width, height, fps, seconds, readAsset,
 *     });
 *
 * Live, from a canvas to a viewer, with no ffmpeg and no readback:
 *
 *     await renderVideo({
 *       source: createCanvasFrameSource(canvas),
 *       encoder: createWebCodecsEncoder(),
 *       sink: createWebSocketSink(socket),
 *       pacing: 'realtime', fps: 30, signal: controller.signal,
 *     });
 */

export type {
  BufferFrame,
  BufferFrameSource,
  CapturedFrame,
  FrameFormat,
  FrameSource,
  ImageFrame,
  ImageFrameSource,
  ShadoVideoEncoder,
  VideoCodec,
  VideoSink,
  VideoSpec,
} from './types';
export {
  renderVideo,
  REALTIME_LOOKAHEAD,
  type Pacing,
  type RenderVideoOptions,
  type VideoResult,
} from './driver';
export { createHttpSink, type HttpSinkOptions } from './http-sink';
export { createWebSocketSink, type WebSocketLike, type WebSocketSinkOptions } from './websocket-sink';
export { createDataChannelSink, type DataChannelLike, type DataChannelSinkOptions } from './datachannel-sink';
export {
  createWebCodecsEncoder,
  isWebCodecsSupported,
  type WebCodecsEncoderOptions,
} from './webcodecs';
export { createCanvasFrameSource, type CanvasFrameSourceOptions, type CanvasLike } from './canvas-source';
export { createMseAppender, isMseSupported, type MseAppender, type MseAppenderOptions } from './mse';
export {
  createScriptFrameSource,
  seekAnimationGroup,
  seekAnimationGroups,
  toVideoScene,
  type SceneContext,
  type SceneScriptModule,
  type SceneSetup,
  type ScriptCapableSession,
  type ScriptFrameSource,
  type ScriptFrameSourceOptions,
  type ScriptLoadOptions,
  type VideoScene,
} from './scene-script';
export {
  createSessionFrameSource,
  orbitCamera,
  type CaptureCapableSession,
  type OrbitOptions,
  type SessionFrameSourceOptions,
} from './session-source';
