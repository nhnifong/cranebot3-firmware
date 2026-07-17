3D models and icons that main.ts and friends reference by path (e.g.
`${VITE_ASSET_BUCKET_URL}/assets/playroom/models/gripper.glb`), checked in
here rather than left as marketing content because the TypeScript reaches
into named objects/meshes *inside* these files — a model and the code that
indexes into it are one unit, and have to change together.

Everything lives under a single `assets/playroom/` folder — not loose in
`assets/` — specifically so a host with its own unrelated assets (like
nf-main-site's marketing photos/videos) can treat "everything this package
needs" as one directory-level rule (one `.gitignore` line, one recursive
copy) instead of maintaining a list of individual filenames that has to be
kept in sync every time a file is added or renamed here.

A host serving this package's app needs everything under `assets/playroom/`
here reachable at `<VITE_ASSET_BUCKET_URL>/assets/playroom/...`:

- **This package's own dev harness** (`npm run dev`) serves it directly —
  `vite.config.ts` points `publicDir` here, and `dev/.env.development` sets
  `VITE_ASSET_BUCKET_URL=""` so paths resolve relatively against it.
- **nf-main-site** copies this directory into its own `public/assets/` at
  build time (see `nf-viz/sync-playroom-assets.sh`) so it rides along with
  the rest of its assets through its existing GCS blue-green bucket sync —
  see that repo's `nf-viz/readme.md`.
- **Any other host** just needs to serve this directory (or a copy of it)
  at `/assets/playroom/` under whatever `VITE_ASSET_BUCKET_URL` it
  configures.
