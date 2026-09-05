import type { ShadoSprite2DMotionConfig } from '@knervous/shado/render';

export type Sprite2DMotionWorkerRequest =
  | {
      type: 'init';
      requestId: number;
      start: number;
      positions: ArrayBuffer;
      config: ShadoSprite2DMotionConfig;
      nowMs: number;
    }
  | {
      type: 'configure';
      requestId: number;
      config: ShadoSprite2DMotionConfig;
      nowMs: number;
    }
  | {
      type: 'step';
      requestId: number;
      nowMs: number;
      dtSeconds: number;
    };

export type Sprite2DMotionWorkerResponse =
  | { type: 'ready' | 'configured'; requestId: number }
  | {
      type: 'result';
      requestId: number;
      start: number;
      positions: ArrayBuffer;
    }
  | { type: 'error'; requestId: number; message: string };

