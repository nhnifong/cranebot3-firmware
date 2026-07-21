import { defineConfig } from 'vite';
import { resolve } from 'path';

// Standalone dev harness so this package is runnable on its own — `npm run
// dev` here gets you the same cockpit UI nf-main-site serves at /playroom,
// pointed at a robot on the LAN, with no cloud account system required (see
// dev/stub-bridge.ts). Consumers like nf-viz bundle src/ directly and never
// go through this config.
export default defineConfig({
  root: 'dev',
  // Vite's default publicDir is <root>/public — override it to the package's
  // own public/ (models, icons) rather than dev/public, since those assets
  // aren't dev-harness-only: they're what any consumer needs to serve at
  // /assets/... (see public/README.md and this package's top-level README).
  publicDir: resolve(__dirname, 'public'),
  build: {
    // Same reasoning as publicDir above: Vite's default outDir is <root>/dist
    // (dev/dist), which is easy to miss since nothing else in this package
    // lives under dev/. Put `npm run build` output at the conventional
    // top-level dist/ instead.
    outDir: resolve(__dirname, 'dist'),
    emptyOutDir: true,
  },
  server: {
    // Listen on 0.0.0.0 (all interfaces), not just localhost, so a phone or
    // other device on the LAN can reach this dev server for testing.
    host: true,
  },
});
