import { Worker } from 'node:worker_threads';
import {
  ShadoEntityVisibilityWorker,
  compileShadoWorld,
  regionsForBounds,
  type ShadoVisibilityWorkerPort,
} from '../src/world';
import type { RegionGrid, ShadoWorldSpatialPackage } from '../src/world';

/**
 * A real worker, running the real source, over the real WASM.
 *
 * The fake-worker tests prove the controller's protocol; they cannot prove
 * that the shipped worker source and the shipped reducer agree with it,
 * because they replace both. This adapter gives the browser worker surface the
 * controller expects -- `postMessage`, `addEventListener('message')`,
 * `addEventListener('error')`, `terminate` -- backed by a Node worker thread
 * executing the exact string the browser would.
 */
function nodeWorkerAdapter(source: string): ShadoVisibilityWorkerPort {
  const shim = `
    const { parentPort } = require('node:worker_threads');
    const self = {
      postMessage: (message, transfer) => parentPort.postMessage(message, transfer),
      onmessage: null,
    };
    parentPort.on('message', (data) => { void self.onmessage({ data }); });
  `;
  const worker = new Worker(`${shim}\n${source}`, { eval: true });
  const messageListeners: ((event: MessageEvent) => void)[] = [];
  const errorListeners: ((event: ErrorEvent) => void)[] = [];
  worker.on('message', (data) => {
    for (const listener of messageListeners) listener({ data } as MessageEvent);
  });
  worker.on('error', (error) => {
    for (const listener of errorListeners) {
      listener({ message: error.message, error } as unknown as ErrorEvent);
    }
  });
  return {
    postMessage: (message, transfer) => worker.postMessage(message, transfer as never),
    addEventListener: (type, listener) => {
      if (type === 'message') messageListeners.push(listener as (event: MessageEvent) => void);
      else errorListeners.push(listener as (event: ErrorEvent) => void);
    },
    terminate: () => void worker.terminate(),
  } as ShadoVisibilityWorkerPort;
}

/** One quad per region, so each region has a cell and a distinct PVS bit. */
function quad(x: number) {
  return {
    name: `quad-${x}`,
    material: 'stone',
    positions: new Float32Array([x, 0, 0, x + 1, 0, 0, x + 1, 1, 0, x, 1, 0]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  };
}

const WIDE_PLANES = new Float32Array([
  1, 0, 0, 4096, -1, 0, 0, 4096, 0, 1, 0, 4096,
  0, -1, 0, 4096, 0, 0, 1, 4096, 0, 0, -1, 4096,
]);

/**
 * The oracle: full-bounds membership and the same policy bits, in plain
 * scalar code with no shared memory, no worker and no WASM.
 *
 * Deliberately a second implementation. Comparing the worker against itself
 * proves only that it is consistent.
 */
function scalarVisible(
  world: ShadoWorldSpatialPackage,
  entities: { x: number; z: number; radius: number; enabled?: boolean; phase?: number }[],
  regionFlags: Uint8Array,
  activePhaseMask = 0xffffffff
): number[] {
  const visibility = world.visibility!;
  const grid: RegionGrid = {
    originX: visibility.originX,
    originZ: visibility.originZ,
    size: visibility.size,
    width: visibility.width,
    height: visibility.height,
  };
  const required = 0x71;
  const scratch = new Uint32Array(64);
  const visible: number[] = [];
  entities.forEach((entity, index) => {
    if (entity.enabled === false) return;
    if (((entity.phase ?? 0xffffffff) & activePhaseMask) === 0) return;
    const membership = regionsForBounds(
      grid,
      entity.x - entity.radius,
      entity.z - entity.radius,
      entity.x + entity.radius,
      entity.z + entity.radius,
      scratch
    );
    if (membership.overflow) {
      visible.push(index);
      return;
    }
    if (!membership.regions.length) return;
    for (const region of membership.regions) {
      if (((regionFlags[region] ?? 0) & required) === required) {
        visible.push(index);
        return;
      }
    }
  });
  return visible;
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

async function waitForResult(worker: ShadoEntityVisibilityWorker, timeoutMs = 4000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const result = worker.acquireLatest();
    if (result) return result;
    await settle();
  }
  throw new Error('the real worker never published a result');
}


import { buildInstancedOccluders } from '../src/world/occluder-instances';
import { compileShadoWorldVisibility } from '../src/world/visibility';
import { computeShadoWorldLayoutHash } from '../src/world/validation';

test('audit: overflow is incorrectly gated by outsideWorldVisible', async () => {
 const world = compileShadoWorld(Array.from({length:80},(_,i)=>quad(i*16)), {name:'audit',tileSize:16,visibilityRegionSize:16,maxClusterTriangles:2});
 const worker = await ShadoEntityVisibilityWorker.create(world,{capacity:8,publishFlags:false,workerFactory:nodeWorkerAdapter});
 try {
 worker.projection.load({count:1,positionX:[640],positionY:[0],positionZ:[0.5],radius:[600]});
 const flags = new Uint8Array(world.visibility!.width*world.visibility!.height).fill(0x71);
 worker.request(WIDE_PLANES,flags,{camera:[640,0,0],outsideWorldVisible:false});
 const got = await waitForResult(worker);
 console.log('AUDIT_OVERFLOW', {regions:flags.length,visible:[...got.visibleIndices],oracle:scalarVisible(world,[{x:640,z:0.5,radius:600}],flags)});
 expect([...got.visibleIndices]).toEqual([]);
 expect(scalarVisible(world,[{x:640,z:0.5,radius:600}],flags)).toEqual([0]);
 } finally {worker.dispose();}
});

test('audit: TLAS ignores zero allocation and cancellation',()=>{
 let polls=0;
 const scene=buildInstancedOccluders([],Array.from({length:1000},()=>({prototype:0,matrix:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]})),{maxBytes:0,stopReason:()=>{polls++;return 'cancelled';}});
 const bytes=scene.instancePrototype.byteLength+scene.instanceInverse.byteLength+scene.instanceBounds.byteLength+scene.instanceMirrored.byteLength+scene.nodeBounds.byteLength+scene.nodeMeta.byteLength+scene.order.byteLength;
 console.log('AUDIT_TLAS',{aborted:scene.aborted,polls,bytes});
 expect(scene.aborted).toBe(null); expect(polls).toBe(0); expect(bytes).toBeGreaterThan(0);
});

test('audit: volume wire bounds and hash',()=>{
 const primitive={name:'floor',material:'stone',positions:new Float32Array([0,0,0,160,0,0,160,0,16,0,0,16]),indices:new Uint32Array([0,1,2,0,2,3])};
 const visibility=compileShadoWorldVisibility({mode:'sampled-occlusion',verticalVolumes:true,bounds:{min:[0,0,0],max:[160,20,16]},regionSize:16,maxDistance:1024,renderCellCenters:[[8,8]],renderCellBounds:[{min:[0,0,0],max:[16,1,16]}],persistentRenderCells:new Uint8Array(1),collisionPrimitives:[primitive]});
 const wire=JSON.parse(JSON.stringify(visibility));
 console.log('AUDIT_VOLUME',{sourceMax: String(visibility.volumes!.maxY[0]),wireMax:wire.volumes.maxY[0]});
 expect(visibility.volumes!.maxY[0]).toBe(Infinity);expect(wire.volumes.maxY[0]).toBe(null);
 const world=compileShadoWorld([quad(0)],{name:'hash',tileSize:16,visibilityRegionSize:16,maxClusterTriangles:2});
 world.visibility=visibility;
 const before=computeShadoWorldLayoutHash(world); visibility.volumes!.minY[0]+=5;
 const after=computeShadoWorldLayoutHash(world);
 console.log('AUDIT_HASH',{before,after});expect(after).toBe(before);
});
