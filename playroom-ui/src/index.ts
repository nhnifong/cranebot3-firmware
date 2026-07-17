// Type-only entry point. The app itself is a side-effecting import — see './app'
// (src/main.ts) — because it isn't a mountable component: it owns the whole
// page (DOM lookups by id, a full-viewport three.js canvas, etc). Import this
// module to get the types needed to implement PlayroomAuthBridge; import
// 'stringman-ui/app' from a second, later <script type="module">
// tag to actually run it, after window.__playroomAuthBridge is set.
export type { PlayroomAuthBridge, RobotInfo, GuestAccess } from './auth-bridge.ts';
