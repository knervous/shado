# Notices

Shado's **source code** is MIT licensed — see [LICENSE](./LICENSE).

The **demo assets** bundled for the sandbox are not, and are not covered by
that MIT grant.

## Ryzom-derived demo assets

These directories contain assets derived from the Ryzom game assets released
by Winch Gate under **CC-BY-SA 3.0**:

| path | used by | contents |
|---|---|---|
| `sandbox/public/shado/hum/` | `/hum-wardrobe` | male humanoid wardrobe: geometry, texture atlas, VAT bakes |
| `sandbox/public/shado/huf/` | `/hum-wardrobe` | female humanoid wardrobe, same shape |
| `sandbox/public/shado/supermesh/` | `/supermesh-scale` | the NM_M humanoid supermesh and its VAT library |

They are derivative works — remeshed, atlas-repacked, and rebaked into vertex
animation textures — so the share-alike term carries: **they remain CC-BY-SA
3.0, with attribution to Winch Gate / the Ryzom project, and any redistribution
of them or of works derived from them must keep those terms.** Using Shado's
code does not oblige you to anything here; bundling these files does.

## Assets that are not in this repository

The sandbox also has routes whose demo content is not redistributable and is
therefore absent from a public clone. They are produced by a private content
pipeline, so the directories are empty rather than broken:

| route | needs | status |
|---|---|---|
| `/` and `/msdf` | `shado/eq-demo` | not bundled |
| `/world`, `/world-editor` | `shado/worlds` | not bundled |
| `/test` | `shado/preprocessed` | not bundled |
| `/ryzom`, `/ryzom-actors` | `shado/ryzom` | not bundled (dev-only routes) |

`/hum-wardrobe` and `/supermesh-scale` are the routes that run from a clean
clone, and they are the ones that demonstrate what the library is for.
