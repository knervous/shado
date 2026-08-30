/**
 * Resolve a promoted object's shared-store texture reference in Node.
 *
 * Promoted GLBs carry `shared/<hash>.ktx2` as their image URI. It is relative
 * on purpose: Babylon's glTF loader rejects any URI containing '..'
 * (`GLTFLoader._ValidateUri`) and otherwise just concatenates `rootUrl + uri`,
 * so a root-absolute path would be mangled rather than resolved. The client
 * maps that prefix onto an HTTP path; there is no server here, so the bytes are
 * read from disk and handed back as a `data:` URL, which Babylon loads without
 * touching the network.
 *
 * The store is located by walking up from this module until a checkout is
 * recognised, so callers need no configuration. `setSharedTextureRoot` exists
 * for a consumer outside this repository layout.
 */
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const STORE_SUFFIX = join('client', 'public', 'eqrequiem', 'textures', 'shared');
const SHARED_URI = /(?:^|\/)shared\/([A-Za-z0-9._-]+\.ktx2)(?:[?#].*)?$/;

let overrideRoot: string | null = null;
let discovered: string | null | undefined;

/** Point the resolver at a store outside the usual checkout layout. */
export function setSharedTextureRoot(directory: string | null): void {
  overrideRoot = directory;
  discovered = undefined;
}

function storeRoot(): string | null {
  if (overrideRoot !== null) return overrideRoot;
  if (discovered !== undefined) return discovered;
  let candidate = dirname(fileURLToPath(import.meta.url));
  for (;;) {
    const store = join(candidate, STORE_SUFFIX);
    if (existsSync(store)) return (discovered = store);
    const parent = dirname(candidate);
    if (parent === candidate) return (discovered = null);
    candidate = parent;
  }
}

/**
 * `preprocessUrlAsync` for the glTF loader. Anything that is not a shared-store
 * reference passes through untouched, and an unresolvable one is returned as-is
 * so the loader reports a missing texture instead of hanging on a promise that
 * never settles.
 */
export async function resolveSharedTextureUri(url: string): Promise<string> {
  const match = SHARED_URI.exec(url);
  if (!match) return url;
  const root = storeRoot();
  if (!root) return url;
  const file = resolve(root, match[1]!);
  if (!file.startsWith(root) || !existsSync(file)) return url;
  const bytes = await readFile(file);
  return `data:image/ktx2;base64,${bytes.toString('base64')}`;
}
