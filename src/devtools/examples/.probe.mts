import { readFile } from 'node:fs/promises';
import { createPreviewSession } from '../session';
const glb = new Uint8Array(await readFile(process.argv[2]));
const decodeImage = async (bytes: Uint8Array) => {
  const sharp = (await import('sharp')).default;
  const { data, info } = await sharp(bytes).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data: new Uint8Array(data) };
};
const session = await createPreviewSession({ width: 256, height: 256, decodeImage });
await session.newScene({ materials: true });
const c = await session.loadGlb(glb, { id: 'rig' });
await session.frameCamera({ view: 'front' });
await new Promise<void>(r => session.scene.executeWhenReady(() => r()));
const mats = c.materials ?? session.scene.materials;
console.log('materials', mats.length);
for (const m of mats.slice(0, 4)) {
  console.log(m.name, '| class', m.getClassName(), '| albedoTexture', !!m.albedoTexture, m.albedoTexture?.isReady?.(),
    '| albedoColor', m.albedoColor?.asArray?.().map((v:number)=>+v.toFixed(2)),
    '| emissive', m.emissiveColor?.asArray?.().map((v:number)=>+v.toFixed(2)),
    '| unlit', m.unlit, '| ambient', session.scene.ambientColor?.asArray?.());
}
console.log('lights', session.scene.lights.map((l:any)=>`${l.name}:${l.intensity}`));
console.log('toneMapping', session.scene.imageProcessingConfiguration?.toneMappingEnabled, session.scene.imageProcessingConfiguration?.exposure);
await session.dispose();
