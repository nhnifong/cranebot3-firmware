3D models and icons that main.ts and friends reference by path (e.g.
`${VITE_ASSET_BUCKET_URL}/assets/models/gripper.glb`), checked in here
rather than left as marketing content because the TypeScript reaches into
named objects/meshes *inside* these files — a model and the code that
indexes into it are one unit, and have to change together.

A host serving this package's app needs everything under `assets/` here
reachable at `<VITE_ASSET_BUCKET_URL>/assets/...`:

- **This package's own dev harness** (`npm run dev`) serves it directly —
  `vite.config.ts` points `publicDir` here, and `dev/.env.development` sets
  `VITE_ASSET_BUCKET_URL=""` so paths resolve relatively against it.
- **nf-main-site** copies this directory into its own `public/assets/` at
  build time (see `nf-viz/package.json`'s `sync-playroom-assets` script) so
  it rides along with the rest of its assets through its existing GCS
  blue-green bucket sync — see that repo's `nf-viz/readme.md`.
- **Any other host** just needs to serve this directory (or a copy of it)
  at `/assets/` under whatever `VITE_ASSET_BUCKET_URL` it configures.
