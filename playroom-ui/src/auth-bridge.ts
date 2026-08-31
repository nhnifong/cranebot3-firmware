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
  // ws:// or wss:// protocol and host of the control plane that minted these
  // credentials (e.g. "wss://neufangled.com"), which the robot stores them
  // under. Only the host knows this: it is the control plane's own telemetry
  // endpoint, not necessarily this page's origin (a dev site on :5173 fronts a
  // control plane on :8080), and the robot cannot infer it because binding
  // happens over the LAN control link. An implementation that omits it leaves
  // the robot guessing from its own --telemetry_env, which is wrong whenever
  // the robot was not started against the control plane that bound it.
  controlPlaneHost?: string;
}

// How many people may drive a robot at once, and where the caller stands.
// Driving means a per-viewer WebRTC feed of every camera, which the media
// gateway can only serve to a handful of people; spectating is an HLS remux that
// fans out cheaply. So a robot shared with everyone can entitle far more people
// to drive than can actually drive, and entitled drivers wait their turn.
export interface DriverSlot {
  // Concurrent drivers this robot allows, and how many are driving now.
  capacity: number;
  in_use: number;
  // Whether the caller is one of them.
  held: boolean;
  // 1-based place in line, or null when not waiting. Always null from
  // apiGetMyAccess: asking about a robot doesn't take a place in line, so a real
  // position only exists once the control socket is open (which is what holds
  // the place). The live position arrives as DriverSlotStatus telemetry.
  queue_position: number | null;
  queue_length: number;
  // Silence after which a held slot is handed to whoever is waiting.
  idle_timeout_seconds: number;
}

export interface MyAccess {
  // What the caller is entitled to: "owner" | "full" | "limited_driver" | "spectator".
  access_level: string;
  // What that entitlement is worth right now. Equal to access_level when a
  // driver slot is available, "spectator" when every slot is taken. Advisory:
  // it predicts what connecting would get you, and the control socket's
  // DriverSlotStatus supersedes it from the moment it opens.
  effective_access_level: string;
  // null when the caller needs no slot (spectators) or when slot state could
  // not be read, in which case treat driving as available and let the control
  // socket decide.
  slot: DriverSlot | null;
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
  // Resolves both what the caller may do and whether they may do it yet.
  apiGetMyAccess(robotId: string, token: string): Promise<MyAccess>;
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
