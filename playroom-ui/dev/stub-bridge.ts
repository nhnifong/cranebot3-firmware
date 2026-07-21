import type { PlayroomAuthBridge } from '../src/auth-bridge.ts';

// Minimal bridge for the standalone dev harness: there's no cloud account
// system here, just a robot on the LAN. LAN mode (and Sim mode) never call
// the token/registry methods below — they connect straight to a websocket —
// so a no-op initAuth is enough to get the app running. isCloudAvailable()
// returning false makes main.ts hide the cloud-only UI (My Robots, Bind to
// account, etc) instead, so the methods below are really just a backstop.
function unimplemented(name: string): never {
  throw new Error(
    `${name}: cloud account features aren't available in this standalone dev ` +
    `harness — use LAN or Sim mode from the landing screen instead.`
  );
}

export const stubAuthBridge: PlayroomAuthBridge = {
  isCloudAvailable: () => false,
  initAuth() {},
  hideSignInUI() {},
  getAuthToken: () => unimplemented('getAuthToken'),
  apiListRobots: () => unimplemented('apiListRobots'),
  apiBindRobot: () => unimplemented('apiBindRobot'),
  apiBindRobotV2: () => unimplemented('apiBindRobotV2'),
  apiGetStreamTicket: () => unimplemented('apiGetStreamTicket'),
  apiUnbindRobot: () => unimplemented('apiUnbindRobot'),
  apiGetMyAccess: () => unimplemented('apiGetMyAccess'),
  apiShareRobot: () => unimplemented('apiShareRobot'),
  apiListAuthorized: () => unimplemented('apiListAuthorized'),
};
