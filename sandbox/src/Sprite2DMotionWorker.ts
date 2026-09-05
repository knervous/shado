/// <reference lib="webworker" />

import {
  ShadoSprite2DMotionKernel,
  type ShadoSprite2DMotionConfig,
} from '@knervous/shado/render/sprite-2d-motion';
import type {
  Sprite2DMotionWorkerRequest,
  Sprite2DMotionWorkerResponse,
} from './Sprite2DMotionWorkerProtocol';

let kernel: ShadoSprite2DMotionKernel | undefined;
let start = 0;

const respond = (message: Sprite2DMotionWorkerResponse, transfer: Transferable[] = []) => {
  self.postMessage(message, { transfer });
};

const configure = (config: ShadoSprite2DMotionConfig, nowMs: number) => {
  if (!kernel) throw new Error('Motion worker has not been initialized');
  kernel.configure(config, nowMs);
};

self.onmessage = async (event: MessageEvent<Sprite2DMotionWorkerRequest>) => {
  const request = event.data;
  try {
    if (request.type === 'init') {
      kernel ??= await ShadoSprite2DMotionKernel.create();
      start = request.start;
      kernel.setPopulation(
        new Float32Array(request.positions),
        request.config,
        request.nowMs,
        request.start
      );
      respond({ type: 'ready', requestId: request.requestId });
      return;
    }
    if (request.type === 'configure') {
      configure(request.config, request.nowMs);
      respond({ type: 'configured', requestId: request.requestId });
      return;
    }
    if (!kernel) throw new Error('Motion worker has not been initialized');
    // WebAssembly.Memory is never shared or transferred. The result is one
    // compact copy whose ownership moves to the main thread.
    const positions = kernel.step(request.nowMs, request.dtSeconds).slice();
    respond({
      type: 'result',
      requestId: request.requestId,
      start,
      positions: positions.buffer,
    }, [positions.buffer]);
  } catch (error) {
    respond({
      type: 'error',
      requestId: request.requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
