/**
 * Which Babylon these dev tools draw with.
 *
 * `@babylonjs/core` is a *peer* dependency of shado, but node resolves a
 * package's own imports against its own `node_modules` first — so a consumer
 * that already owns a Babylon (the eltania client owns 9.18) still gets shado's
 * dev copy (9.16) here, and ends up with two engine graphs in one process.
 * Two graphs is not a version-skew nuisance, it is a hard break: separate
 * `ShaderStore`s, separate `Effect` registries, and a `Scene` the host's own
 * `Mesh` fails `instanceof` against.
 *
 * Every Babylon import in this folder is dynamic, so one hook is enough to point
 * them all at the host's copy:
 *
 *   useBabylonRuntime((specifier) => import(specifier));   // from the host's module
 *
 * Called from a module inside the host package, that arrow resolves against the
 * host's `node_modules`. Left uninstalled, the default keeps today's behaviour.
 */

export type BabylonModuleImporter = (specifier: string) => Promise<any>;

const ownImporter: BabylonModuleImporter = (specifier) =>
  import(/* @vite-ignore */ specifier);

let importer: BabylonModuleImporter = ownImporter;

/**
 * Routes every Babylon import in the dev tools through `next`. Pass `null` to
 * restore shado's own resolution.
 */
export function useBabylonRuntime(next: BabylonModuleImporter | null): void {
  importer = next ?? ownImporter;
}

/** True once a host has claimed the Babylon runtime. */
export function hasHostBabylonRuntime(): boolean {
  return importer !== ownImporter;
}

/** Imports a Babylon module by full specifier, e.g. `@babylonjs/core/scene.js`. */
export function babylonImport(specifier: string): Promise<any> {
  return importer(specifier);
}
