# stringman-ui

Browser UI for controlling a Stringman robot: 3D telemetry
visualization (three.js), live camera feeds, gamepad/touch drive controls,
and LeRobot-assisted recording. This is the same UI served at `/playroom` on
neufangled.com, extracted here so it ships as part of the open-source robot
software rather than the private website repo, and it's a fully standalone,
runnable app on its own, with no account/cloud system required.

## Running it standalone

How to open the UI pointed at a robot on your local network.

**Prerequisites**: Node.js 20+. Either works on both macOS and Linux:

- via [nvm](https://github.com/nvm-sh/nvm) (lets you pin a Node version per
  project, recommended if you're not sure):

      curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
      source ~/.bashrc   # or ~/.zshrc
      nvm install 20
      nvm use 20

- or your system package manager, macOS: `brew install node`; Linux
  (Debian/Ubuntu): `sudo apt install nodejs npm`; other distros' equivalents.

**Setup and run**, from this directory (`playroom-ui/`),

    npm install
    npm run dev

Then open the URL Vite prints (`http://localhost:5173` by default). You'll
land on the connection-mode screen, pick **LAN only mode** to connect
straight to a robot running with `stringman-headless`.

## Developing this package alongside nf-main-site

If you're changing this package's code and want nf-main-site to pick up
those changes without publishing, check out both repos as siblings (e.g.
`~/cranebot3-firmware` next to `~/nf-main-site`) and point nf-viz's
`package.json` at a local path:

    "stringman-ui": "file:../../cranebot3-firmware/playroom-ui"

`npm install` in `nf-viz` symlinks it into `node_modules`, so edits here are
picked up immediately by `npm run dev`/`npm run build` over there, no
publish step needed. Re-run `npm run proto` in this directory whenever
`../src/nf_robot/protos/*.proto` changes; nf-viz doesn't know to do that for
you.

A `file:` dependency only resolves on a machine with both repos checked out
side by side, though, it won't resolve in CI or a Docker build. Those need
a real published version, see Publishing below.

## Publishing

This package is published on the public npm registry as `stringman-ui`.

One-time setup, on any machine you'll publish from:

    npm login          # opens a browser to authenticate; needs an npmjs.com account

Before releasing a version:

    npm run verify-against-nf-viz:selftest

If it's all good, then bump the version and publish.

    npm version patch   # or minor/major
    npm publish

## Architecture

A few decisions here aren't obvious from the code alone, so: why this is
shaped the way it is.

**It ships source, not a build.** `package.json` points straight at
`src/*.ts`, there's no compile step producing a `dist/` for consumers.
Whatever bundles this (nf-viz's Vite config, or your own) compiles it as
part of its own build. That's a deliberate trade: it means
`import.meta.env.VITE_ASSET_BUCKET_URL`, used throughout for model/image
URLs, resolves against *the host's* build-time environment, not this
package's. A prebuilt bundle would have baked in whichever bucket URL
happened to be set when *this* package was published, which is wrong for
every host except the one that built it.

**It's not a mountable component, it owns the page.** `main.ts` looks up
around 200 DOM ids directly and appends a full-viewport `<canvas>`; there's
no props/slots API. The markup those ids live in is `src/app-shell.html`, a
raw HTML fragment (not a Vite entry page) that any host injects into its own
`<body>`, see `dev/bridge.ts` for the reference implementation, using
Vite's `?raw` import. The fragment and `main.ts` are really one artifact
split across two files for editor convenience; they aren't meant to vary
independently, which is also why the shell lives in this package instead of
being left for each host to maintain its own copy.

**There's an auth bridge because account/cloud logic isn't robot software.**
nf-main-site layers a whole SaaS system on top of this UI, Firebase login,
a multi-tenant robot registry, sharing robots with other accounts, HTTP
routes like `/listrobots` and `/bind` that only exist on nf-main-site's own
backend. None of that is meaningful to someone who clones this repo and
points the UI at a robot on their LAN; baking it in would mean the "open
source robot software" also drags along a private company's account system.
So `main.ts` never talks to auth/account logic directly, it calls a
`PlayroomAuthBridge` (defined in `src/auth-bridge.ts`) that the host installs
at `window.__playroomAuthBridge` before the app boots. nf-main-site's
implementation wraps its real Firebase/API calls; `dev/stub-bridge.ts` (used
by `npm run dev` here) is a no-op, since LAN/Sim mode connect straight to a
websocket and never need a token. The bridge also exposes
`isCloudAvailable()`, which `main.ts` checks to hide cloud-only UI (My
Robots, Bind to account, Share access) rather than show buttons that would
just error when there's no account system behind them.

Wiring one up needs two separate `<script type="module">` tags, in this
order:

```html
<script type="module" src="./bridge.ts"></script>
<script type="module" src="./app-entry.ts"></script>
```

```ts
// bridge.ts
import 'stringman-ui/style.css';
import 'stringman-ui/mobile.css';
window.__playroomAuthBridge = { /* implementation */ };
```

```ts
// app-entry.ts
import 'stringman-ui/app';
```

They have to be separate files, not one file with the assignment sandwiched
between two `import` statements, all of a module's own imports evaluate
before any of its top-level statements run, so the assignment wouldn't
happen in time if `./app` were imported from the same file as the bridge
setup. Separate `<script type="module">` tags execute in document order
instead, which gives the ordering the app actually needs.

**3D models and icons are checked in here, not left as website content.**
`main.ts`/`objects/*.ts` index named meshes inside the `.glb` files in
`public/assets/playroom/models/`, a model and the code that reaches into it
by name are one unit, the same reasoning as the app shell above, so they
live next to the TypeScript instead of in nf-main-site alongside marketing
photos and product videos (see `public/README.md`, which also explains why
everything's namespaced under one `playroom/` folder rather than loose in
`assets/`). They're expected to be reachable at
`<VITE_ASSET_BUCKET_URL>/assets/playroom/...`. Standalone (this package's
own `npm run dev`), that's just `public/` served directly with no bucket
configured. nf-viz instead copies this directory into its own `public/` at
build time (`sync-playroom-assets.sh`), so the files ride along through
nf-main-site's existing asset pipeline, its own `public/` gets synced to a
GCS bucket on
deploy, rather than this package needing to know anything about buckets or
blue-green deploys at all.
