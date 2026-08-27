import type {
  ShadoWorldAuthoringDocument,
  ShadoWorldAudioEmitter,
  ShadoWorldCompiledAudioEmitter,
  ShadoWorldObjectStamp,
  WorldVec3,
} from './types';
import { transformStampOffset } from './point-lights';

/** Resolves standalone and inherited object emitters into canonical Babylon world space. */
export function resolveShadoWorldAudioEmitters(
  document: Pick<ShadoWorldAuthoringDocument, 'environment' | 'objects'>
): ShadoWorldCompiledAudioEmitter[] {
  const standalone = document.environment.audioEmitters.map(emitter => ({
    id: emitter.id,
    sourceKind: 'standalone' as const,
    enabled: true,
    source: emitter.source,
    position: [...emitter.position] as WorldVec3,
    range: emitter.range,
    volume: emitter.volume,
    loop: emitter.loop,
    metadata: { ...emitter.metadata },
  }));
  const prototypes = new Map(document.objects.prototypes.map(prototype => [prototype.id, prototype]));
  const attached = document.objects.stamps.flatMap(stamp => {
    const emitter = stamp.audio ?? prototypes.get(stamp.prototype)?.audio;
    return emitter ? [resolveObjectAudioEmitter(stamp, emitter)] : [];
  });
  return [...standalone, ...attached];
}

function resolveObjectAudioEmitter(
  stamp: ShadoWorldObjectStamp,
  emitter: ShadoWorldAudioEmitter
): ShadoWorldCompiledAudioEmitter {
  const offset = transformStampOffset(stamp, emitter.offset);
  return {
    id: `${stamp.id}:audio`,
    sourceKind: 'object',
    ownerStamp: stamp.id,
    enabled: stamp.enabled && emitter.enabled,
    source: emitter.source,
    position: [
      stamp.position[0] + offset[0],
      stamp.position[1] + offset[1],
      stamp.position[2] + offset[2],
    ],
    range: emitter.range,
    volume: emitter.volume,
    loop: emitter.loop,
    metadata: { ...emitter.metadata },
  };
}
