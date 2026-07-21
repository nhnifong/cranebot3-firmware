// The playroom app is rendered by different hosts with different account
// systems: nf-main-site's cloud control plane (Firebase auth + a per-account
// robot registry) versus a robot owner running this package standalone
// against a robot on their LAN. Rather than bake nf-main-site's cloud
// account logic into this open-source package, the host supplies an
// implementation of PlayroomAuthBridge on `window.__playroomAuthBridge`
// before loading the './app' entry point (see README for wiring examples).

export interface RobotInfo {
  nickname: string;
  robotid: string;
  online: boolean;
  role: string;
  // "owner" | "full" | "limited_driver" | "spectator"
  access_level?: string;
}

// One shared guest and the level they were granted.
export interface GuestAccess {
  email: string;
  access_level: string;
}

// The identity issued by /bindv2 when a brand-new (previously anonymous) robot
// is bound to an account: a freshly minted robot id and the secret key the robot
// must present to publish authenticated telemetry. The key is returned exactly
// once and must be handed to the robot for storage.
export interface BindV2Result {
  robotId: string;
  key: string;
}

export interface PlayroomAuthBridge {
  // True for a real cloud-account implementation (nf-main-site's), false for
  // a stub like dev/stub-bridge.ts. main.ts uses this to hide cloud-only UI
  // (the "My Robots" landing button/header link, "Bind to account", etc)
  // rather than showing controls that would just fail when used standalone.
  isCloudAvailable(): boolean;
  initAuth(onUserChange?: (user: unknown) => void): void;
  hideSignInUI(): void;
  getAuthToken(): Promise<string>;
  apiListRobots(token: string): Promise<RobotInfo[]>;
  apiBindRobot(robotId: string, nickname: string, token: string): Promise<void>;
  // Binds a brand-new (previously anonymous) robot: the server mints an id and
  // key, and the implementation hands the key to the robot for storage.
  apiBindRobotV2(nickname: string, token: string): Promise<BindV2Result>;
  apiGetStreamTicket(robotId: string, token: string): Promise<string>;
  apiUnbindRobot(robotId: string, token: string): Promise<void>;
  apiGetMyAccess(robotId: string, token: string): Promise<string>;
  apiShareRobot(robotId: string, email: string, accessLevel: string, token: string): Promise<void>;
  apiListAuthorized(robotId: string, token: string): Promise<GuestAccess[]>;
}

declare global {
  interface Window {
    __playroomAuthBridge?: PlayroomAuthBridge;
  }
}

export function getAuthBridge(): PlayroomAuthBridge {
  const bridge = window.__playroomAuthBridge;
  if (!bridge) {
    throw new Error(
      "No auth bridge found on window.__playroomAuthBridge. The host page must set it before " +
      "importing 'stringman-ui/app' — see this package's README."
    );
  }
  return bridge;
}
