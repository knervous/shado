import fsSync from 'fs'
import fs from 'fs/promises'
import path from 'path'

/**
 * Dev-server API for the Ryzom asset library pane.
 *
 * The converted library is ~5.7 GB across ~1,900 GLBs, so nothing is copied
 * into public/: assets stream straight out of assets/ryzom/converted. The
 * catalog is built from the sidecars once and cached, because reading 2,400
 * JSON files per request would make the pane feel broken.
 *
 *   GET  /__ryzom/catalog            index of every converted asset
 *   GET  /__ryzom/asset/<cat>/<name> the GLB itself
 *   POST /__ryzom/review             persist a review decision
 *
 * Review decisions are written to assets/ryzom/candidates.json, the same
 * promotion ledger tools/ryzom/build_asset_review.ts maintains, so triage done
 * here and triage done in the static review page share one source of truth.
 */

type CatalogEntry = {
  id: string
  name: string
  category: string
  source: string
  triangles: number
  meshes: number
  materials: number
  textures: number
  animations: number
  skinned: boolean
  joints?: number
  bytes: number
  size: [number, number, number] | null
  warnings: number
  unresolvedTextures: boolean
  reviewStatus: string
  eltaniaMapping: string | null
  notes: string[]
  /** Set once the asset has been imported into Eltania as an object. */
  promotedId: string | null
  promotedAt: string | null
  /**
   * Provenance, for the external corpora. Core art has no family and is always
   * promotable; external art may be blocked, and the pane has to be able to say
   * so before someone clicks Promote and collects a 400.
   */
  sourceFamily: string | null
  licenseConfidence: string | null
  promotionBlocked: string | null
  styleReview: string | null
  /** Owner decision allowing promotion despite unverified provenance. */
  provenanceOverride: string | null
}

/** Libra's standalone dev API, per serverjs/src/libra/dev-env.ts. */
const LIBRA_API = process.env.LIBRA_API_URL ?? 'http://127.0.0.1:8082/libra'

/**
 * Eltania object ids are lowercase [a-z0-9_-]. Ryzom names are already close;
 * this only has to guarantee the shape and a stable `ryz-` namespace so a
 * promoted asset is never confused with hand-authored Eltania content.
 */
function eltaniaObjectId(name: string): string {
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 100)
  return `ryz-${slug || 'object'}`
}

export function ryzomLibraryPlugin(repoRoot: string) {
  const convertedRoot = path.resolve(repoRoot, 'assets/ryzom/converted')
  const candidatesPath = path.resolve(repoRoot, 'assets/ryzom/candidates.json')

  let cache: { builtAt: number; entries: CatalogEntry[] } | null = null

  async function readCandidates(): Promise<Record<string, any>> {
    const raw = await fs.readFile(candidatesPath, 'utf8').then(JSON.parse).catch(() => null)
    return raw?.assets ?? {}
  }

  async function buildCatalog(): Promise<CatalogEntry[]> {
    const candidates = await readCandidates()
    const entries: CatalogEntry[] = []

    async function walk(dir: string): Promise<void> {
      const listing = await fs.readdir(dir, { withFileTypes: true }).catch(() => [])
      for (const item of listing) {
        const full = path.join(dir, item.name)
        if (item.isDirectory()) {
          await walk(full)
          continue
        }
        if (!item.name.endsWith('.asset.json')) continue

        const sidecar = await fs.readFile(full, 'utf8').then(JSON.parse).catch(() => null)
        if (!sidecar) continue
        const stem = item.name.slice(0, -'.asset.json'.length)
        const category = path.basename(path.dirname(full))
        const glb = full.replace(/\.asset\.json$/, '.glb')
        const bytes = await fs.stat(glb).then((s) => s.size, () => 0)
        // An empty conversion has no GLB; it stays in the catalog so the
        // reviewer can see it exists and why, but it cannot be previewed.
        if (!bytes) continue

        const stats = sidecar.stats ?? {}
        const bbox = stats.bbox
        const ledger = candidates[sidecar.source] ?? {}
        entries.push({
          id: `${category}/${stem}`,
          name: sidecar.originalName ?? stem,
          category,
          source: sidecar.source ?? '',
          triangles: stats.triangles ?? 0,
          meshes: stats.meshes ?? 0,
          materials: stats.materials ?? 0,
          textures: stats.textures ?? 0,
          animations: stats.animations ?? 0,
          skinned: !!stats.skinned,
          joints: stats.joints,
          bytes,
          size: bbox
            ? (bbox.max.map((v: number, i: number) => +(v - bbox.min[i]).toFixed(2)) as [number, number, number])
            : null,
          warnings: (sidecar.warnings ?? []).length,
          unresolvedTextures: (sidecar.warnings ?? []).some((w: string) =>
            w.startsWith('unresolved textures:')),
          reviewStatus: ledger.reviewStatus ?? sidecar.reviewStatus ?? 'unreviewed',
          eltaniaMapping: ledger.eltaniaMapping ?? null,
          notes: sidecar.notes ?? [],
          promotedId: ledger.promotedId ?? null,
          promotedAt: ledger.promotedAt ?? null,
          sourceFamily: ledger.sourceFamily ?? sidecar.sourceFamily ?? null,
          licenseConfidence: ledger.licenseConfidence ?? sidecar.licenseConfidence ?? null,
          promotionBlocked: ledger.promotionBlocked ?? null,
          styleReview: ledger.styleReview ?? sidecar.styleReview ?? null,
          provenanceOverride: ledger.provenanceOverride ?? sidecar.provenanceOverride?.reason ?? null,
        })
      }
    }

    await walk(convertedRoot)
    entries.sort((a, b) =>
      a.category.localeCompare(b.category) || a.name.localeCompare(b.name))
    return entries
  }

  function readBody(request: any, limit = 1024 * 1024): Promise<string> {
    return new Promise((resolve, reject) => {
      const chunks: Buffer[] = []
      let bytes = 0
      request.on('data', (chunk: Buffer) => {
        bytes += chunk.byteLength
        if (bytes > limit) reject(new Error('request body too large'))
        else chunks.push(chunk)
      })
      request.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
      request.on('error', reject)
    })
  }

  return {
    name: 'ryzom-library',
    configureServer(server: any) {
      // Walking ~2,400 sidecars takes seconds; do it once at startup so the
      // first visit to the pane is not a cold 16-second wait.
      void (async () => {
        try {
          cache = { builtAt: Date.now(), entries: await buildCatalog() }
          server.config.logger.info(`  ryzom library: ${cache.entries.length} assets indexed`)
        } catch {
          // The pane reports a missing library itself; a failed warm-up is not
          // worth taking the dev server down for.
        }
      })()

      server.middlewares.use('/__ryzom/catalog', async (request: any, response: any) => {
        try {
          const refresh = String(request.url ?? '').includes('refresh=1')
          // The ledger is rewritten by the seeder whenever the library actually
          // changes, so its mtime is the cheap staleness signal — one stat per
          // request instead of re-walking ~4,800 sidecars, and the pane picks up
          // a conversion run without anyone restarting the dev server.
          const ledgerChanged = await fs.stat(candidatesPath)
            .then((st) => !!cache && st.mtimeMs > cache.builtAt)
            .catch(() => false)
          if (refresh || ledgerChanged || !cache) {
            cache = { builtAt: Date.now(), entries: await buildCatalog() }
          }
          const byCategory: Record<string, number> = {}
          for (const entry of cache.entries) {
            byCategory[entry.category] = (byCategory[entry.category] ?? 0) + 1
          }
          response.setHeader('Content-Type', 'application/json')
          response.end(JSON.stringify({
            builtAt: cache.builtAt,
            total: cache.entries.length,
            byCategory,
            entries: cache.entries,
          }))
        } catch (error) {
          response.statusCode = 500
          response.end(JSON.stringify({ error: String(error) }))
        }
      })

      server.middlewares.use('/__ryzom/asset/', (request: any, response: any) => {
        // Resolve inside the converted root and verify containment, so a
        // traversal in the URL cannot reach the rest of the repo.
        const relative = decodeURIComponent(String(request.url ?? '').replace(/^\//, '').split('?')[0])
        const resolved = path.resolve(convertedRoot, relative)
        if (!resolved.startsWith(convertedRoot + path.sep) || !resolved.endsWith('.glb')) {
          response.statusCode = 400
          response.end('bad asset path')
          return
        }
        let stat: fsSync.Stats
        try {
          stat = fsSync.statSync(resolved)
        } catch {
          response.statusCode = 404
          response.end('not found')
          return
        }
        response.setHeader('Content-Type', 'model/gltf-binary')
        response.setHeader('Content-Length', stat.size)
        response.setHeader('Cache-Control', 'no-cache')
        fsSync.createReadStream(resolved).pipe(response)
      })

      server.middlewares.use('/__ryzom/promote', async (request: any, response: any) => {
        if (request.method !== 'POST') {
          response.statusCode = 405
          response.end('POST required')
          return
        }
        try {
          const { id: assetId, boundsRadius } = JSON.parse(await readBody(request))
          const entry = cache?.entries.find((e) => e.id === assetId)
          if (!entry) throw new Error(`unknown asset '${assetId}'`)

          // External art whose provenance is still yellow may be reviewed and
          // previewed, but it must never reach eltania-ready — the seeder marks
          // those `promotionBlocked`, and this is where that mark has to bite.
          // See docs/EXTERNAL_RYZOM_ART_INTAKE_PLAN.md §2.
          const ledgerEntry = (await readCandidates())[entry.source] ?? {}
          if (ledgerEntry.promotionBlocked) {
            throw new Error(
              `'${assetId}' is blocked from promotion (${ledgerEntry.promotionBlocked}). `
              + 'Resolve its provenance before promoting.',
            )
          }

          const glbPath = path.resolve(convertedRoot, `${assetId}.glb`)
          if (!glbPath.startsWith(convertedRoot + path.sep)) throw new Error('bad asset path')
          const glb = await fs.readFile(glbPath)

          // Promotion goes through Libra's own import endpoint rather than
          // writing the manifest here. That call also inspects the geometry,
          // records a libra_asset_versions row and writes an audit entry —
          // reimplementing it locally would silently skip all three.
          const objectId = eltaniaObjectId(entry.name)
          // Half the largest extent is the enclosing radius; Libra defaults to
          // 32, which is wildly wrong for a 0.5 m prop.
          const radius = boundsRadius
            ?? (entry.size ? Math.max(0.5, +(Math.max(...entry.size) / 2).toFixed(2)) : 32)
          const query = new URLSearchParams({
            id: objectId,
            name: entry.name,
            boundsRadius: String(radius),
          })
          const importResponse = await fetch(`${LIBRA_API}/assets/objects/import?${query}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/octet-stream' },
            body: glb,
          }).catch((error: unknown) => {
            throw new Error(
              `Libra API unreachable at ${LIBRA_API} (${String(error)}). `
              + 'Start it with: npm --prefix serverjs run libra:api:dev',
            )
          })
          const payload = await importResponse.json().catch(() => ({}))
          if (!importResponse.ok) {
            throw new Error(`Libra import failed (${importResponse.status}): ${JSON.stringify(payload)}`)
          }

          const raw = await fs.readFile(candidatesPath, 'utf8').then(JSON.parse).catch(() => ({ assets: {} }))
          raw.assets ??= {}
          raw.assets[entry.source] = {
            ...(raw.assets[entry.source] ?? {}),
            reviewStatus: 'eltania-ready',
            eltaniaMapping: objectId,
            promotedId: objectId,
            promotedAt: new Date().toISOString(),
          }
          raw.updatedAt = new Date().toISOString()
          await fs.writeFile(candidatesPath, JSON.stringify(raw, null, 2) + '\n')

          entry.reviewStatus = 'eltania-ready'
          entry.eltaniaMapping = objectId
          entry.promotedId = objectId
          entry.promotedAt = raw.assets[entry.source].promotedAt

          response.setHeader('Content-Type', 'application/json')
          response.end(JSON.stringify({ ok: true, objectId, boundsRadius: radius, asset: payload }))
        } catch (error) {
          response.statusCode = 400
          response.end(JSON.stringify({ error: error instanceof Error ? error.message : String(error) }))
        }
      })

      server.middlewares.use('/__ryzom/review', async (request: any, response: any) => {
        if (request.method !== 'POST') {
          response.statusCode = 405
          response.end('POST required')
          return
        }
        try {
          const { source, reviewStatus, eltaniaMapping } = JSON.parse(await readBody(request))
          if (!source) throw new Error('source is required')

          const raw = await fs.readFile(candidatesPath, 'utf8').then(JSON.parse).catch(() => ({ assets: {} }))
          raw.assets ??= {}
          raw.assets[source] = {
            ...(raw.assets[source] ?? {}),
            reviewStatus: reviewStatus ?? raw.assets[source]?.reviewStatus ?? 'unreviewed',
            eltaniaMapping: eltaniaMapping ?? raw.assets[source]?.eltaniaMapping ?? null,
            reviewedAt: new Date().toISOString(),
          }
          raw.updatedAt = new Date().toISOString()
          await fs.writeFile(candidatesPath, JSON.stringify(raw, null, 2) + '\n')

          // Keep the in-memory catalog in step so the pane does not need a reload.
          const entry = cache?.entries.find((e) => e.source === source)
          if (entry) {
            entry.reviewStatus = raw.assets[source].reviewStatus
            entry.eltaniaMapping = raw.assets[source].eltaniaMapping
          }
          response.setHeader('Content-Type', 'application/json')
          response.end(JSON.stringify({ ok: true, entry: raw.assets[source] }))
        } catch (error) {
          response.statusCode = 400
          response.end(JSON.stringify({ error: String(error) }))
        }
      })
    },
  }
}
