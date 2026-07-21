import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
// Import the generated protobuf classes
import { nf } from './generated/proto_bundle.js';
import { Gripper } from './objects/gripper.ts';
import { ArpAnchor } from './objects/arp_anchor.ts';
import { Eyelet } from './objects/eyelet.ts';
import { Gantry } from './objects/gantry.ts';
import { Cable } from './objects/cable.ts';
import { FloorProjection } from './objects/floor_projection.ts';
import { DynamicRoom } from './objects/dynamic_room.ts'
import { SightingsManager } from './objects/sightings_manager.ts'
import { VideoFeed } from './ui/video_feed.ts'
import { GamepadController } from './ui/gamepad.ts'
import { MobileShell } from './ui/mobile.ts'
import { TargetListManager } from './ui/target_list_manager.ts'
import { Say, Listen } from './utils.ts';
import { isTutorialMode, maybeStartTutorial } from './tutorial.ts';

import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/examples/jsm/postprocessing/GTAOPass.js';
import { OutlinePass } from 'three/examples/jsm/postprocessing/OutlinePass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';

// Auth Manager — supplied by the host page on window.__playroomAuthBridge,
// since account/robot-registry logic differs between hosts (see auth-bridge.ts).
import { getAuthBridge, type RobotInfo } from './auth-bridge.ts';
const AuthManager = getAuthBridge();

// --- GLOBAL VARIABLES ---
const DEFAULT_CAM_TILT = 26.0; // degrees — matches the standard tilt adapter

// Debug toggle: show wireframe frustum helpers for the anchor cameras used in floor-projection raycasting
const SHOW_ANCHOR_CAMERA_FRUSTUMS = false;

const urlParams: URLSearchParams = new URLSearchParams(window.location.search);
// If robotid is set in URL, we force cloud login. Otherwise we start the landing UI.
let currentRobotId: string | null = urlParams.get('robotid'); 
let detectedRobotId: string | null = null; // Stored from incoming telemetry
let swingCancellationEnabled = false;
let autoTargetEnabled = false;
let tensionRegEnabled = true;
let userRobots: RobotInfo[] = [];

// Calibration tracking variables
let isFullCalibrationActive = false;
let anchorsSeeingOriginCard: number[] = [];
let lastCalibratedStatus: nf.common.CalibratedStatus = nf.common.CalibratedStatus.CALIBRATEDSTATUS_UNSET;

// Gripper Sensor State
let currentGripperPressure = 0;
let currentGripperTargetForce: number | null = null;

// LeRobot Session State
let isLeRobotSessionActive = false;
let isLeRobotStarting = false; // Track pending startup
let leRobotState: nf.common.LerobotStatus = nf.common.LerobotStatus.LEROBOTSTATUS_NA;
let numEpisodesRecorded = 0;
let datasetEpCount = 0;
let hfRepoId = "username/my-lerobot-dataset";
let policyRepoId = "username/my-policy";
let lerobotError: string | null = null;
let episodeStartTime: number | null = null;
let sentFinalizeCommand = false;
let episodesUntilCheckpoint: number | null = null;

// Motion perspective modes
const perspViewport = 0; 
const perspPerson = 1;
const perspTop = 2;
const perspBottom = 3;
const perspGripper = 5;
let currentPerspective: number = perspGripper; // control the initial selection

// Scene Setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x444444);

const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(2, 2, 5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
// Cap device pixel ratio so high-DPI phones don't try to render at 3× cost.
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

// Initialize Composer
const composer = new EffectComposer(renderer);

// Render Pass: Draws the scene normally first
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);

// GTAO Pass: Adds the shading
const gtaoPass = new GTAOPass(scene, camera, window.innerWidth, window.innerHeight);
gtaoPass.setSize(window.innerWidth, window.innerHeight);
composer.addPass(gtaoPass);

// Outline Pass for prominent hover selection
const outlinePass = new OutlinePass(new THREE.Vector2(window.innerWidth, window.innerHeight), scene, camera);
outlinePass.edgeStrength = 5.0; // Prominence of the outline
outlinePass.edgeThickness = 2.0; // Thickness of the outline
outlinePass.edgeGlow = 0.0;
outlinePass.visibleEdgeColor.set(0xffffff); // Bright white outline
outlinePass.hiddenEdgeColor.set(0x888888); // Dimmer if occluded
composer.addPass(outlinePass);

// Output Pass: Tone mapping and sRGB conversion
const outputPass = new OutputPass();
composer.addPass(outputPass);

composer.setSize(window.innerWidth, window.innerHeight);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Skip GTAO on mobile — too expensive on phone GPUs.
if (MobileShell.shouldEnable()) {
  gtaoPass.enabled = false;
}

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.5, 0);
controls.update();

// Lights
const ambientLight = new THREE.AmbientLight(0x808080, 2);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 3);
dirLight.position.set(2, 5, 2);
scene.add(dirLight);

// Geometry
const room = new DynamicRoom(scene);

// Input handler
const gamepad = new GamepadController();

// Shared corners array: 2 anchors (indices 0-1) + 2 eyelets (indices 2-3), the fixed
// Arpeggio topology.
const acoords = [
  { x: 2.5,  y: -2.5, rotZ: -Math.PI / 4 },
  { x: 2.5,  y: 2.5,  rotZ: -3 * Math.PI / 4 },
  { x: -2.5, y: 2.5,  rotZ: 3 * Math.PI / 4 },
  { x: -2.5, y: -2.5, rotZ: Math.PI / 4 },
];

const corners: (ArpAnchor | Eyelet)[] = acoords.map((ac, i) => {
  const corner: ArpAnchor | Eyelet = i < 2 ? new ArpAnchor(scene, room) : new Eyelet(scene, room);
  corner.setPose(nf.common.Pose.create({
    position: { x: ac.x, y: ac.y, z: 3 },
    rotation: { x: 0, y: 0, z: ac.rotZ }
  }));
  return corner;
});

// Gantry
const gantry = new Gantry(scene);

// Gripper
const gripper = new Gripper(scene);

// Main support Cables
const cables: Cable[] = [0, 1, 2, 3].map((i) => {
  const cable = new Cable(scene);
  cable.update(corners[i].grommet_pos, gantry.position, 0.0)
  return cable;
});

// Wall-Routing Cables for Arpeggio
const wallCables: Cable[] = [];

// Visual gantry factor
const gantryVisualCube = new THREE.Mesh(
  new THREE.BoxGeometry(0.1, 0.1, 0.1),
  new THREE.MeshStandardMaterial({ color: 0xff0000, transparent: true, opacity: 0.5 }));
gantryVisualCube.quaternion.setFromAxisAngle( new THREE.Vector3( 0, 1, 0 ), Math.PI / 4 );
gantryVisualCube.position.set(-0.2, 1.5, 0);
scene.add(gantryVisualCube);

// Visual hang factor
const gantryHangCube = new THREE.Mesh(
  new THREE.BoxGeometry(0.1, 0.1, 0.1),
  new THREE.MeshStandardMaterial({ color: 0x0099ff, transparent: true, opacity: 0.5 }));
gantryHangCube.quaternion.setFromAxisAngle( new THREE.Vector3( 0, 1, 0 ), Math.PI / 4 );
gantryHangCube.position.set(0.2, 1.5, 0);
scene.add(gantryHangCube);

// gantry sightings manager
const sightingsManager = new SightingsManager(scene);
// laser rangefinder feedback re-uses sightings manager class
const laserReadings = new SightingsManager(scene, {color: 0xFF0000, radius: 0.005});

const targetListManager = new TargetListManager();

// gamepad needs a reference
gamepad.targetListManager = targetListManager;

// Video feed managers
const firstOverheadVideo = new VideoFeed(document.getElementById('firstOverhead')!, targetListManager);
const secondOverheadVideo = new VideoFeed(document.getElementById('secondOverhead')!, targetListManager);
const gripperVideo = new VideoFeed(document.getElementById('gripper')!);
const overheadVideofeeds = [firstOverheadVideo, secondOverheadVideo];
const floorProjection = new FloorProjection(scene, 5, 0.001);

// Debug visualization: wireframe frustums for the anchor cameras used in floor-projection
// raycasting, so misalignment between the virtual camera and the video feed is visible.
const cameraHelpers: (THREE.CameraHelper | null)[] = [];

// Listen for hover changes in the manager to trigger repaints in video feeds
targetListManager.onTargetHover = () => {
  firstOverheadVideo.refresh();
  secondOverheadVideo.refresh();
};

targetListManager.onTargetSelect = () => {
  firstOverheadVideo.refresh();
  secondOverheadVideo.refresh();
};

// ------ Authorization and Connection Logic ------

let socket: WebSocket;
let isLanMode = false;
let isSimMode = false; // Track simulator mode

// The caller's effective access level for the connected robot. 'owner' until
// resolved (and always effectively owner in LAN/sim, which have no cloud auth).
let currentAccessLevel: string = 'owner';
// Spectators send nothing; everyone else may send (the backend filters
// limited_driver, so the UI need only fully block spectators here).
function canSendControl(): boolean {
  return currentAccessLevel !== 'spectator';
}

// LeRobot recording/eval is a full-access feature; hidden for limited_driver and
// spectator. (updateLeRobotUI also consults this so it can't re-show the button.)
function canUseLeRobot(): boolean {
  return currentAccessLevel === 'owner' || currentAccessLevel === 'full';
}

// Command-issuing menu items and the levels allowed to use each. Items not listed
// (e.g. Toggle Log Panel, Get Stream Ticket) are view/utility and stay available.
// The backend is the real gate (access_policy.py); this just avoids dead buttons.
const CONTROL_MENU_ITEMS: Record<string, string[]> = {
  'action-pick-drop':         ['owner', 'full'],
  'action-half-cal':          ['owner', 'full', 'limited_driver'],
  'action-record-park':       ['owner', 'full'],
  'action-grasp':             ['owner', 'full', 'limited_driver'],
  'action-full-cal':          ['owner', 'full'],
  'action-dataset':           ['owner', 'full'],
  'action-collect-images':    ['owner', 'full'],
  'action-update-firmware':   ['owner', 'full'],
  'action-disable-torque':    ['owner', 'full'],
  'action-enable-torque':     ['owner', 'full'],
  'action-toggle-tension-reg':['owner', 'full'],
  'action-debug-send':        ['owner', 'full'],
};

// Grey out (via the existing `disabled` class, which the menu handlers respect)
// the controls the current access level can't use, and show a level badge when
// the caller isn't the owner.
function applyAccessLevelUI() {
  for (const [id, allowed] of Object.entries(CONTROL_MENU_ITEMS)) {
    const el = document.getElementById(id);
    if (!el) continue;
    const ok = allowed.includes(currentAccessLevel);
    el.classList.toggle('disabled', !ok);
    if (!ok) el.title = 'Not available at your access level';
  }

  // LeRobot recording/eval is a full-access feature.
  document.getElementById('btn-header-lerobot')?.classList.toggle('hidden', !canUseLeRobot());

  updateAccessBadge();
}

// Small indicator of the current access level, shown just to the right of the
// online/offline status in the footer. Hidden for owners.
function updateAccessBadge() {
  const statusText = document.getElementById('status-text');
  let badge = document.getElementById('access-level-badge');
  if (!badge) {
    if (!statusText?.parentElement) return; // status line not in DOM yet
    badge = document.createElement('span');
    badge.id = 'access-level-badge';
    badge.style.cssText = 'margin-left:8px;background:rgba(0,0,0,0.55);color:#fff;' +
      'padding:2px 8px;border-radius:10px;font-size:11px;text-transform:capitalize;';
    statusText.insertAdjacentElement('afterend', badge);
  }
  badge.textContent = currentAccessLevel.replace('_', ' ');
  badge.style.display = currentAccessLevel === 'owner' ? 'none' : 'inline-block';
}

// Helpers to sync URL
function setUrlParam(id: string | null) {
  const url = new URL(window.location.href);
  if (id) {
    url.searchParams.set('robotid', id);
  } else {
    url.searchParams.delete('robotid');
  }
  window.history.pushState({}, '', url);
}

// ============================================================
//  Log Panel — bottom-of-screen strip toggled from the
//  Calibration & Maintenance menu.
// ============================================================
const MAX_LOG_LINES = 1000;
const logLines: string[] = [];

// Append forwarded log lines from telemetry, capping the buffer at
// MAX_LOG_LINES and keeping the panel scrolled to the bottom if the
// user was already there.
function handleLogs(data: nf.telemetry.ILogs) {
  if (!data.line || data.line.length === 0) return;
  logLines.push(...data.line);
  if (logLines.length > MAX_LOG_LINES) {
    logLines.splice(0, logLines.length - MAX_LOG_LINES);
  }

  const content = document.getElementById('log-panel-content');
  if (!content) return;
  const atBottom = content.scrollHeight - content.scrollTop - content.clientHeight < 4;
  content.textContent = logLines.join('\n');
  if (atBottom) {
    content.scrollTop = content.scrollHeight;
  }
}

function showLogPanel() {
  document.getElementById('ui-layer')?.classList.add('logs-open');
  // Ask the robot to start forwarding debug logs over telemetry.
  simpleCommand(nf.control.Command.COMMAND_DEBUG_LOG_OVER_T);
}

function hideLogPanel() {
  document.getElementById('ui-layer')?.classList.remove('logs-open');
}

function toggleLogPanel() {
  const uiLayer = document.getElementById('ui-layer');
  if (!uiLayer) return;
  if (uiLayer.classList.contains('logs-open')) {
    hideLogPanel();
  } else {
    showLogPanel();
  }
}

// Entry Point
function initApp() {
  // No cloud account system (e.g. running this package standalone outside
  // nf-main-site) -> hide cloud-only UI instead of showing controls that
  // would just fail when used.
  if (!AuthManager.isCloudAvailable()) {
    document.getElementById('btn-cloud-mode')?.classList.add('hidden');
    document.getElementById('header-link-my-robots')?.classList.add('hidden');
  }

  // If a user is already logged in via Firebase session,
  // trigger a silent fetch of their robots list so that bound check works in LAN mode.
  AuthManager.initAuth(async (user) => {
    const btn = document.getElementById('btn-cloud-mode');
    if (btn && user) {
      btn.innerHTML = `My Robots<div class="landing-hint">Manage your cloud-connected robots</div>`;
      
      try {
        const token = await AuthManager.getAuthToken();
        userRobots = await AuthManager.apiListRobots(token);
        // If the robot already detected (LAN), update button now that we have the list
        refreshRunMenuAuth();
      } catch (e) {
        console.warn("Silent robot list fetch failed:", e);
      }
    }
  });

  // Bind all landing and panel buttons unconditionally
  document.getElementById('btn-lan-mode')?.addEventListener('click', startLanFlow);
  document.getElementById('btn-sim-mode')?.addEventListener('click', startSimFlow);
  document.getElementById('btn-cloud-mode')?.addEventListener('click', handleCloudLogin);
  
  // Bind binding panel buttons
  document.getElementById('btn-cancel-bind')?.addEventListener('click', () => {
    document.getElementById('landing-layer')?.classList.add('hidden');
    document.getElementById('bind-robot-panel')?.classList.add('hidden');
    document.getElementById('landing-options')?.classList.remove('hidden');
  });
  
  document.getElementById('btn-confirm-bind')?.addEventListener('click', executeBind);

  // The /tutorial alias always starts in LAN mode and overlays the guided tour.
  if (isTutorialMode()) {
    startLanFlow();
    maybeStartTutorial();
  } else if (currentRobotId === 'lan') {
    startLanFlow();
  } else if (currentRobotId === 'sim') {
    startSimFlow();
  } else if (currentRobotId) {
    // URL param set -> Force cloud flow
    updateRobotIdUI(currentRobotId);
    startCloudFlow(currentRobotId);
  } else if (!AuthManager.isCloudAvailable()) {
    // No cloud account system -> LAN is the only real option (Sim mode is
    // nf-main-site's own FastAPI endpoint, not available standalone), so
    // skip the landing panel instead of offering choices that don't work.
    startLanFlow();
  } else {
    // No URL param -> Show landing page
    const landing = document.getElementById('landing-layer');
    if (landing) landing.classList.remove('hidden');
  }
}

// LAN Mode Flow
function startLanFlow() {
  isLanMode = true;
  setUrlParam('lan');
  document.getElementById('landing-layer')?.classList.add('hidden');
  updateRobotIdUI("Localhost");
  
  // Show the Bind button since we are in LAN mode
  document.getElementById('action-bind')?.classList.remove('hidden');
  
  // Where the control websocket lives depends on how this page itself was
  // loaded. When neufangled.com hosts the page, "LAN mode" means the classic
  // case of a browser and robot on the very same machine, so localhost is
  // always right there. When this bundle is self-hosted by stringman-headless
  // (--serve_ui) there's no cloud account bridge (isCloudAvailable() is
  // false), and the page was fetched from the robot's own LAN address — so
  // the robot is wherever this page came from, not necessarily localhost.
  // Deriving the host from window.location also keeps this same-origin with
  // the page, which avoids mixed-content blocking (an https:// neufangled.com
  // page can't open ws:// to a non-loopback LAN host; a self-hosted http://
  // page connecting back to its own origin has no such restriction).
  const target = AuthManager.isCloudAvailable()
    ? "ws://localhost:4245"
    : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}:4245`;
  connect(target);
}

// Sim Mode Flow
function startSimFlow() {
  isSimMode = true;
  setUrlParam('sim');
  document.getElementById('landing-layer')?.classList.add('hidden');
  document.getElementById('btn-header-lerobot')?.classList.add('hidden');
  document.getElementById('lerobot-overlay')?.classList.add('hidden');
  updateRobotIdUI("Simulated Robot");

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/simulated/arp`;
  
  connect(wsUrl);
}

// Cloud Mode Flow
async function handleCloudLogin() {
  AuthManager.initAuth();
  try {
    const token = await AuthManager.getAuthToken();
    document.getElementById('landing-options')?.classList.add('hidden');
    document.getElementById('robot-list-panel')?.classList.remove('hidden');
    userRobots = await AuthManager.apiListRobots(token);
    renderRobotList(userRobots);
  } catch (error) { console.error("Login failed:", error); }
}

async function startCloudFlow(robotId: string) {
  AuthManager.initAuth();
  try {
    const token = await AuthManager.getAuthToken();

    // Resolve our access level before connecting so the UI starts in the right
    // state. A force_access URL param overrides it for local testing.
    try {
      currentAccessLevel = await AuthManager.apiGetMyAccess(robotId, token);
    } catch (e) {
      console.warn('Could not resolve access level, assuming owner:', e);
      currentAccessLevel = 'owner';
    }
    applyAccessLevelUI();

    const ticket = await AuthManager.apiGetStreamTicket(robotId, token);
    document.getElementById('landing-layer')?.classList.add('hidden');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let url = `${protocol}//${window.location.host}/control/${robotId}?ticket=${encodeURIComponent(ticket)}`;
    connect(url);
  } catch (error) { console.error("Cloud connection failed:", error); }
}

// --- Binding Logic ---

async function handleBindAction() {
  // A previously-anonymous robot has no id to detect: apiBindRobotV2 mints one on bind, so
  // (unlike the old apiBindRobot flow) we no longer need a robot id from telemetry here.

  // Force Login (if not already)
  AuthManager.initAuth();
  try {
    await AuthManager.getAuthToken();
    
    // Show Bind Panel
    const landing = document.getElementById('landing-layer');
    const options = document.getElementById('landing-options');
    const bindPanel = document.getElementById('bind-robot-panel');
    
    if (landing && options && bindPanel) {
      landing.classList.remove('hidden');
      options.classList.add('hidden');
      bindPanel.classList.remove('hidden');
    }
  } catch (error) {
    console.error("Login required for binding:", error);
    alert("You must log in to bind a robot.");
  }
}

async function executeBind() {
  const nickname = (document.getElementById('bind-nickname') as HTMLInputElement)?.value || "Stringman";
  try {
    const token = await AuthManager.getAuthToken();
    await new Promise(r => setTimeout(r, 100));
    // Binding a previously-anonymous robot needs no client-supplied id: the server
    // mints an id + key and returns them. We then hand the key to the robot over
    // its live (LAN) control link via a RelayCreds control message, so it can
    // store the creds and reconnect to the cloud authenticated as its new id.
    const { robotId, key } = await AuthManager.apiBindRobotV2(nickname, token);
    sendControl([nf.control.ControlItem.create({
      addRelayCreds: nf.common.RelayCreds.create({ robotId, key }),
    })]);
    document.getElementById('landing-layer')?.classList.add('hidden');
    document.getElementById('bind-robot-panel')?.classList.add('hidden');
    showPopup({message: "Success! Robot bound to your account."});
    // Refresh robots list
    userRobots = await AuthManager.apiListRobots(token);
    refreshRunMenuAuth();
  } catch (error) { console.error(error); }
}

function refreshRunMenuAuth() {
  const authMenuItem = document.getElementById('action-robot-auth');
  if (!authMenuItem) return;

  if (!AuthManager.isCloudAvailable()) {
    authMenuItem.classList.add('hidden');
    return;
  }

  const robotInfo = userRobots.find(b => b.robotid === detectedRobotId);

  if (!robotInfo) {
    // Robot is not bound to this user
    authMenuItem.classList.remove('hidden');
    authMenuItem.textContent = "Bind to account";
    authMenuItem.onclick = handleBindAction;
  } else if (robotInfo.role === 'owner') {
    // User is the owner
    authMenuItem.classList.remove('hidden');
    authMenuItem.textContent = "Share access";
    authMenuItem.onclick = handleShareAction;
  } else {
    // User is a guest
    authMenuItem.classList.add('hidden');
  }
}

// --- Unbind Logic ---

// Callback storage for the confirmation dialog
let pendingConfirmAction: (() => void) | null = null;

function initConfirmDialog() {
  const overlay = document.getElementById('confirm-overlay');
  const yesBtn = document.getElementById('confirm-yes');
  const noBtn = document.getElementById('confirm-no');

  if (yesBtn) {
    yesBtn.addEventListener('click', () => {
      if (pendingConfirmAction) pendingConfirmAction();
      overlay?.classList.add('hidden');
      pendingConfirmAction = null;
    });
  }

  if (noBtn) {
    noBtn.addEventListener('click', () => {
      overlay?.classList.add('hidden');
      pendingConfirmAction = null;
    });
  }
}
initConfirmDialog();

function handleUnbindClick(e: Event, robotId: string) {
  e.stopPropagation(); // prevent selecting the robot
  
  const overlay = document.getElementById('confirm-overlay');
  const msg = document.getElementById('confirm-message');
  
  if (overlay && msg) {
    msg.textContent = `Unlink robot ${robotId} from your account?`;
    overlay.classList.remove('hidden');
    
    pendingConfirmAction = () => executeUnbind(robotId);
  }
}

async function executeUnbind(robotId: string) {
  try {
    const token = await AuthManager.getAuthToken();
    
    await AuthManager.apiUnbindRobot(robotId, token);

    // Refresh list
    const robots = await AuthManager.apiListRobots(token);
    renderRobotList(robots);

  } catch (error) {
    console.error("Unbind error:", error);
    alert("Failed to unlink robot.");
  }
}

// --- List Logic ---

function renderRobotList(robots: RobotInfo[]) {
  const container = document.getElementById('robot-list-container');
  if (!container) return;
  container.innerHTML = '';

  if (robots.length === 0) {
    container.innerHTML = '<div style="padding:20px; color:#aaa">No robots found. You must first connect in LAN mode and select Bind from the run menu.</div>';
    return;
  }

  robots.forEach(bot => {
    const el = document.createElement('div');
    el.className = 'robot-list-item';
    
    // Create inner structure manually to bind events cleanly
    const infoDiv = document.createElement('div');
    infoDiv.style.flexGrow = '1';
    infoDiv.innerHTML = `
        <div class="robot-name">${bot.nickname || 'Unnamed Robot'}</div>
        <div class="robot-id">${bot.robotid}</div>
    `;
    
    const statusDiv = document.createElement('div');
    statusDiv.className = `status-badge ${bot.online ? 'status-online' : ''}`;
    statusDiv.textContent = bot.online ? 'ONLINE' : 'OFFLINE';

    // Unbind button with SVG icon (Broken Link)
    const unbindBtn = document.createElement('button');
    unbindBtn.className = 'btn-unbind';
    unbindBtn.title = 'Unlink Robot';
    unbindBtn.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
        <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
        <line x1="14" y1="11" x2="10" y2="13" stroke="currentColor" stroke-width="2" stroke-dasharray="2"></line> <!-- Simulating broken link with dash -->
      </svg>
    `;
    
    unbindBtn.onclick = (e) => handleUnbindClick(e, bot.robotid);

    el.appendChild(infoDiv);
    el.appendChild(statusDiv);
    el.appendChild(unbindBtn);

    el.onclick = () => {
      currentRobotId = bot.robotid;
      updateRobotIdUI(currentRobotId);
      // Update URL without reload
      const newUrl = new URL(window.location.href);
      newUrl.searchParams.set('robotid', currentRobotId);
      window.history.pushState({}, '', newUrl);
      
      startCloudFlow(currentRobotId);
    };
    container.appendChild(el);
  });
}

function updateRobotIdUI(id: string) {
  const el = document.getElementById('ui-robot-id');
  if (el) el.textContent = id;
}

// --- WebSocket Logic ---

function disconnect() {
  if (socket) {
    // Disable the onclose handler so it doesn't try to reconnect
    socket.onclose = null;
    socket.close();
  }
  updateOnlineStatus(false);
  
  // Reset modes
  isLanMode = false;
  isSimMode = false;
  updateRobotIdUI("Unknown");
  setUrlParam(null);
}

function connect(wsUrl: string) {
  if (socket) {
    socket.onclose = null;
    socket.close();
  }
  socket = new WebSocket(wsUrl);
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => {
    if (isLanMode || isSimMode) updateOnlineStatus(true);
  };

  socket.onmessage = (event: MessageEvent) => {
    try {
      const data = new Uint8Array(event.data as ArrayBuffer);
      const batch = nf.telemetry.TelemetryBatchUpdate.decode(data);

      // Capture ID from telemetry stream if available
      if (batch.robotId && batch.robotId !== detectedRobotId) {
        detectedRobotId = batch.robotId;
        console.log("Telemetry from Robot ID:", detectedRobotId);
        if (isLanMode) {
          updateRobotIdUI(detectedRobotId);
        }
        refreshRunMenuAuth();
      }

      for (const update of batch.updates) {
        if (update.newAnchorPoses) handleNewAnchorPoses(update.newAnchorPoses);
        else if (update.posEstimate) handlePosEstimate(update.posEstimate);
        else if (update.posFactorsDebug) handlePosFactorsDebug(update.posFactorsDebug);
        else if (update.lastCommandedVel) handleLastCommandedVel(update.lastCommandedVel);
        else if (update.vidStats) handleVidStats(update.vidStats);
        else if (update.componentConnStatus) handleComponentConnStatus(update.componentConnStatus);
        else if (update.targetList) handleTargetList(update.targetList);
        else if (update.gantrySightings) sightingsManager.handleSightings(update.gantrySightings);
        else if (update.popMessage) showPopup(update.popMessage);
        else if (update.namedPosition) handleNamedPosition(update.namedPosition);
        else if (update.videoReady) handleVideoReady(update.videoReady);
        else if (update.uplinkStatus) handleUplinkStatus(update.uplinkStatus);
        else if (update.gripSensors) handleGripSensors(update.gripSensors);
        else if (update.gripCamPreditions) gripperVideo.setGripperPredictions(update.gripCamPreditions);
        else if (update.operationProgress) handleOperationProgress(update.operationProgress);
        else if (update.swingCancellationState) handleSwingCancellationState(update.swingCancellationState);
        else if (update.tensionRegulationState) handleTensionRegulationState(update.tensionRegulationState);
        else if (update.autoTargetingState) handleAutoTargetingState(update.autoTargetingState);
        else if (update.taskStatus) handleTaskStatus(update.taskStatus);
        else if (update.visibilityStates) handleVisibilityStates(update.visibilityStates);
        else if (update.logs) handleLogs(update.logs);
        else if (update.episodeControl) handleEpisodeControl(update.episodeControl);
      }
    } catch (err) {
      console.error("Decode error:", err);
    }
  };

  socket.onclose = (event) => {
    if (event.code === 1008) {
      console.error("Access revoked. Closing video connections.");
      firstOverheadVideo.setOffline();
      secondOverheadVideo.setOffline();
      gripperVideo.setOffline();
      floorProjection.setOffline();
    } else {
      console.warn("Disconnected. Retrying in 2s...");
      updateOnlineStatus(false);
      setTimeout(() => connect(wsUrl), 2000);
    }
  };
  
  socket.onerror = (err) => {
    console.error("WebSocket Error:", err);
  };
}

function handleEpisodeControl(data: nf.common.IEpisodeControl) {
  if (data.status) {
    const status = data.status;
    isLeRobotStarting = false; // Once we get a status update, we are no longer in the "Starting..." phase
    const oldState = leRobotState;
    isLeRobotSessionActive = (
      status.status !== nf.common.LerobotStatus.LEROBOTSTATUS_REC_ALL_COMPLETE
      && status.status !== nf.common.LerobotStatus.LEROBOTSTATUS_EVAL_ALL_COMPLETE
      && status.status !== nf.common.LerobotStatus.LEROBOTSTATUS_ERROR
      && status.status !== nf.common.LerobotStatus.LEROBOTSTATUS_NA
      && status.status !== null
      );
    leRobotState = status.status ?? nf.common.LerobotStatus.LEROBOTSTATUS_NA;
    numEpisodesRecorded = status.sessionEpNumber ?? 0;
    if (status.datasetEpCount) datasetEpCount = status.datasetEpCount;
    if (status.datasetRepoId) {
      hfRepoId = status.datasetRepoId;
      const datasetInput = document.getElementById('lerobot-input-dataset-name') as HTMLInputElement | null;
      if (datasetInput) datasetInput.value = hfRepoId;
    }
    if (status.policyRepoId) {
      policyRepoId = status.policyRepoId;
      const policyInput = document.getElementById('lerobot-input-policy') as HTMLInputElement | null;
      if (policyInput) policyInput.value = policyRepoId;
    }
    episodesUntilCheckpoint = status.episodesUntilCheckpoint ?? null;
    lerobotError = status.error ?? null;

    if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_RECORDING) {
      Say(`Starting episode ${numEpisodesRecorded}`);
    } else if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_REC_READY) {
      Say(`Ready`);
    } else if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_REC_CHECKPOINT) {
      Say(`Checkpoint uploading, please wait`);
    } else if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_REC_PROCESSING && sentFinalizeCommand) {
      Say(`Recording Ended, Processing video`);
    } else if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_REC_ALL_COMPLETE) {
      Say(`Complete`);
    } else if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_REC_EP_ABANDONED) {
      Say(`Abandoned Episode`);
    }

    // Handle timer reset/start logic
    const becameActive = (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_RECORDING || leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_EVAL_ACTIVE);
    const wasActive = (oldState === nf.common.LerobotStatus.LEROBOTSTATUS_RECORDING || oldState === nf.common.LerobotStatus.LEROBOTSTATUS_EVAL_ACTIVE);
    
    if (becameActive && !wasActive) {
      episodeStartTime = Date.now();
    } else if (!becameActive) {
      episodeStartTime = null;
    }

    updateLeRobotUI();
  }
}

// Operation names that have reached 100% completion at least once this session.
const completedOperations: Record<string, boolean> = {};

/** True if the named operation has ever reached 100% completion this session. */
export function hasOperationCompleted(name: string): boolean {
  return completedOperations[name] === true;
}

function handleOperationProgress(data: nf.telemetry.IOperationProgress) {
  let percentComplete = (data.percentComplete ?? 0);

  const container = document.getElementById('op-progress-container');
  const nameEl = document.getElementById('op-name');
  const actionEl = document.getElementById('op-action');
  const fillEl = document.getElementById('op-bar-fill');

  // Track if we are inside the 'Full Calibration' operation to manage monkey emoji visibility
  if (data.name?.toLowerCase().includes('alibration')) {
    isFullCalibrationActive = percentComplete < 100;
    if (percentComplete < 0.1) {
      anchorsSeeingOriginCard = [];
    }
  } else if (percentComplete >= 100) {
    isFullCalibrationActive = false;
  }
  updateFloatingLabelsText();

  if (!container || !nameEl || !actionEl || !fillEl) return;

  // 100% Logic
  if (percentComplete >= 100) {
    if (data.name) completedOperations[data.name] = true;
    container.classList.add('hidden');
    // Use data.name and data.currentAction for the popup
    showPopup({ message: `${data.name ?? 'Operation'} \n${data.currentAction ?? ''}` });
    return;
  }

  // Show and Update
  container.classList.remove('hidden');
  nameEl.textContent = data.name ?? 'Operation';
  actionEl.textContent = data.currentAction ?? '';
  fillEl.style.width = `${Math.max(0, Math.min(100, percentComplete))}%`;
}

// Start the app
initApp();

function sendGamepad() {
  const items = gamepad.checkInputsAndCreateControlItems(leRobotState);
  if (items.length > 0) {
    sendControl(items);
  }
}

// --- Floating Labels Updater ---
function handleVisibilityStates(data: nf.telemetry.IVisibilityStates) {
  anchorsSeeingOriginCard = data.anchorsSeeingOriginCard || [];
  updateFloatingLabelsText();
}

function updateFloatingLabelsText() {
  for (let i = 0; i < corners.length; i++) {
    const labelEl = document.getElementById(`label-anchor-${i}`);
    if (labelEl) {
      if (corners[i] instanceof Eyelet) {
        labelEl.textContent = `Eyelet ${i}`;
      } else {
        if (isFullCalibrationActive && !anchorsSeeingOriginCard.includes(i)) {
          labelEl.textContent = `Anchor ${i} 🙈`;
        } else {
          labelEl.textContent = `Anchor ${i}`;
        }
      }
    }
  }
}

const down15cm = new THREE.Vector3(0, -0.15, 0);

function updateFloatingLabels() {
  const halfWidth = window.innerWidth / 2;
  const halfHeight = window.innerHeight / 2;
  const pos = new THREE.Vector3();

    // Anchors and Eyelets
  for (let i = 0; i < corners.length; i++) {
    const corner = corners[i];
    const labelEl = document.getElementById(`label-anchor-${i}`);

    if (corner.grommet_pos && labelEl) {
      pos.copy(corner.grommet_pos);

            // Project 3D position to 2D screen coordinates
      pos.project(camera);

            // Hide label if corner is behind the camera
      if (pos.z > 1) {
        labelEl.classList.add('hidden');
        continue;
      }

      labelEl.classList.remove('hidden');

            // Convert to CSS coordinates
      const x = (pos.x * halfWidth) + halfWidth;
      const y = -(pos.y * halfHeight) + halfHeight;

            // Apply translation. Subtracting 100% and 10px from Y dynamically ensures the
            // bottom-left corner of the div tracks the anchor position precisely while hovering above it.
      labelEl.style.transform = `translate(${x}px, calc(${y}px - 100% - 10px))`;
    }
  }

    // Gripper
  const gripperLabelEl = document.getElementById('label-gripper-force');
  if (gripper.grommet_pos && gripperLabelEl) {
    pos.copy(gripper.grommet_pos);
    pos.add(down15cm);
    pos.project(camera);

    gripperLabelEl.classList.remove('hidden');
    const x = (pos.x * halfWidth) + halfWidth;
    const y = -(pos.y * halfHeight) + halfHeight;

        // Positioned centered below the gripper so it's near the fingers.
    gripperLabelEl.style.transform = `translate(${x - 25}px, ${y}px)`;
    const fillEl = document.getElementById('gripper-force-fill');
    const targetEl = document.getElementById('gripper-force-target');

    if (fillEl) {
      fillEl.style.width = `${Math.max(0, Math.min(100, currentGripperPressure * 100))}%`;
    }

    if (targetEl) {
      if (currentGripperTargetForce !== null) {
        targetEl.classList.remove('hidden');
        targetEl.style.left = `${Math.max(0, Math.min(100, currentGripperTargetForce * 100))}%`;
      } else {
        targetEl.classList.add('hidden');
      }
    }
  }
}

function updateLerobotEpisodeTimer() {
  const timerEl = document.getElementById('lerobot-timer');
  if (episodeStartTime !== null && timerEl) {
    const elapsed = (Date.now() - episodeStartTime) / 1000;
    timerEl.textContent = elapsed.toFixed(1) + 's';
    timerEl.classList.remove('hidden');
  } else if (timerEl) {
    timerEl.classList.add('hidden');
  }
}

// --- Render Loop ---
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  sightingsManager.update();
  laserReadings.update();
  floorProjection.update();
  for (const helper of cameraHelpers) helper?.update();
  updateLerobotEpisodeTimer();
  composer.render();
  sendGamepad();
  updateFloatingLabels();
}

animate();

// Resize Handler
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
  gtaoPass.setSize(window.innerWidth, window.innerHeight);
  outlinePass.setSize(window.innerWidth, window.innerHeight);
  if (expandedFeedId) {
    const panelRight = document.getElementById('panel-right');
    const feed = document.getElementById(expandedFeedId);
    if (panelRight && feed) {
      const rect = panelRight.getBoundingClientRect();
      feed.style.left = `${rect.left}px`;
      feed.style.top = `${rect.top}px`;
      feed.style.width = `${rect.width}px`;
      feed.style.height = `${rect.height}px`;
    }
  }
});

//  ===== telemetry update handlers =====

// Reposition the anchors and eyelets. Corners are always
// [ArpAnchor, ArpAnchor, Eyelet, Eyelet]s
function handleNewAnchorPoses(data: nf.telemetry.IAnchorPoses) {
  if (wallCables.length === 0) {
    wallCables.push(new Cable(scene));
    wallCables.push(new Cable(scene));
  }
  if ((data as any).tilt) {
    lastTiltAngles = (data as any).tilt;
    for (let i = 0; i < 2 && i < lastTiltAngles.length; i++) {
      const camera = (corners[i] as ArpAnchor).camera;
      if (camera) {
        applyAnchorCamTilt(i, camera, lastTiltAngles[i]);
      }
    }
  }
  if (data.swingLatency) {
    const slider = document.getElementById('swing-latency-slider') as HTMLInputElement | null;
    const valueDisplay = document.getElementById('swing-latency-value');
    let val = data.swingLatency.toFixed(2);
    slider!.value = val;
    valueDisplay!.textContent = val + 's';
  }

  if (data.calibrated != null) {
    lastCalibratedStatus = data.calibrated;
  }

  updateComponentStatusUI();

  if (data.poses && data.poses.length >= 2) {
    (corners[0] as ArpAnchor).setPose(data.poses[0]);
    (corners[1] as ArpAnchor).setPose(data.poses[1]);
  }
  if (data.eyelets && data.eyelets.length >= 2) {
    (corners[2] as Eyelet).setPosition(data.eyelets[0]);
    (corners[3] as Eyelet).setPosition(data.eyelets[1]);
  }

    // Update wall cables (0 routes to 3, 1 routes to 2)
  Promise.all([corners[0].ready, corners[2].ready]).then(() => {
    wallCables[0].update((corners[0] as ArpAnchor).extra_grommet_pos, corners[2].grommet_pos, 0.0);
    const cam0 = (corners[0] as ArpAnchor).camera!;
    firstOverheadVideo.setVirtualCamera(cam0);
    attachCameraHelper(0, cam0);
  });
  Promise.all([corners[1].ready, corners[3].ready]).then(() => {
    wallCables[1].update((corners[1] as ArpAnchor).extra_grommet_pos, corners[3].grommet_pos, 0.0);
    const cam1 = (corners[1] as ArpAnchor).camera!;
    secondOverheadVideo.setVirtualCamera(cam1);
    attachCameraHelper(1, cam1);
  });

  updateFloatingLabelsText();

  // redraw cables
  cables.forEach((cable, i) => {
    cable.update(corners[i].grommet_pos, gantry.position, 0.0);
  });
}

function handlePosEstimate(data: nf.telemetry.IPositionEstimate) {
  if (data.gantryPosition) {
    gantry.setPosition(data.gantryPosition);
    // redraw cables
    cables.forEach((cable, i) => {
      cable.update(corners[i].grommet_pos, gantry.position, data.slack![i] ? 0.2 : 0.0);
    });
    // Inform input controller so it can calculate input orbits
    gamepad.setRobotPosition(data.gantryPosition.x!, data.gantryPosition.y!);
    [firstOverheadVideo, secondOverheadVideo, gripperVideo].forEach(f => f.setGantryPosition(data.gantryPosition!));
  }

  if (data.gripperPose) {
    // Draw velocity indicator
    gripper.setPose(data.gripperPose);
  }
}

function handlePosFactorsDebug(data: nf.telemetry.IPositionFactors) {
  if (data.visualPos) {
    gantryVisualCube.position.set(data.visualPos.x!, data.visualPos.z!, -data.visualPos.y!);
  }
  if (data.hangingPos) {
    gantryHangCube.position.set(data.hangingPos.x!, data.hangingPos.z!, -data.hangingPos.y!);
  }
}

function handleLastCommandedVel(data: nf.telemetry.ICommandedVelocity) {
  if (data.velocity) {
    // Draw commanded velocity indicator
    gantry.setVelocity(data.velocity);
  }
}

function handleVidStats(data: nf.telemetry.IVidStats) {
  // Get Elements
  const dpsEl = document.getElementById('detectionRate');
  const latEl = document.getElementById('videoLatency');
  const fpsEl = document.getElementById('videoFramerate');

  // Format & Update elements
  if (dpsEl) {
    dpsEl.textContent = (data.detectionRate ?? 0).toFixed(1);
  }
  if (latEl) {
    latEl.textContent = (data.videoLatency ?? 0).toFixed(2) + ' s';
  }
  if (fpsEl) {
    fpsEl.textContent = (data.videoFramerate ?? 0).toFixed(1) + ' fps';
  }
}

// Track state of all components since updates arrive one by one
interface ComponentState {
  name: string;
  type: 'Anchor' | 'Gripper';
  status: number;
  ip: string;
  errorMessage?: string;
  temp?: number;
  motorTorque?: nf.telemetry.MotorTorque;
}
const componentStates = new Map<string, ComponentState>();

export function isFullyConnected(): boolean {
  const expectedAnchors = 2;
  let anchorConnected = 0;
  let gripperConnected = 0;

  componentStates.forEach(comp => {
    const connected = comp.status === nf.telemetry.ConnStatus.CONNSTATUS_CONNECTED;
    if (comp.type === 'Anchor' && connected) anchorConnected++;
    if (comp.type === 'Gripper' && connected) gripperConnected++;
  });

  return anchorConnected >= expectedAnchors && gripperConnected >= 1;
}

export function isFullyCalibrated(): boolean {
  return lastCalibratedStatus === nf.common.CalibratedStatus.CALIBRATEDSTATUS_FULLY_CALIBRATED;
}

function updateComponentStatusUI() {
  updateSwingToggleAvailability();

  let anchorCount = 2;
  let anchorConnected = 0;
  let gripperCount = 1;
  let gripperConnected = 0;

  componentStates.forEach(comp => {
    const isConnected = comp.status === nf.telemetry.ConnStatus.CONNSTATUS_CONNECTED;
    
    if (comp.type === 'Anchor') {
      if (isConnected) anchorConnected++;
    } else if (comp.type === 'Gripper') {
      if (isConnected) gripperConnected++;
    }
  });

  const btnText = document.getElementById('component-status-text');
  if (btnText) {
    btnText.textContent = `Anchors (${anchorConnected}/${anchorCount}) Gripper (${gripperConnected}/${gripperCount})`;
  }

  const menu = document.getElementById('component-menu');
  if (menu) {
    menu.innerHTML = '';
    
    const sortedKeys = Array.from(componentStates.keys()).sort();

    if (sortedKeys.length === 0) {
      menu.innerHTML = '<div class="menu-item disabled">Waiting for telemetry...</div>';
      return;
    }

    sortedKeys.forEach(key => {
      const comp = componentStates.get(key)!;
      const row = document.createElement('div');
      row.className = 'comp-row';

      const isConnected = comp.status === nf.telemetry.ConnStatus.CONNSTATUS_CONNECTED;
      const statusClass = isConnected ? 'comp-status-good' : 'comp-status-bad';
      const statusText = isConnected ? 'Connected' : 'Disconnected';
      const ip = comp.ip || 'Unknown IP';
      
      row.innerHTML = `
        <div class="comp-header">
          <span>${comp.name}</span>
          <span class="${statusClass}">${statusText}</span>
        </div>
        <div class="comp-details">
          <span class="cd-ip-copyable" title="Copy IP address"><span class="comp-ip-text">${ip}</span><svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor"><path d="M4 2a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V2zm2-1a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H6z"/><path d="M2 5a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-1H8v1H2V6h1V5H2z"/></svg></span>
          <span></span>
        </div>
      `;
      const ipEl = row.querySelector<HTMLElement>('.cd-ip-copyable');
      ipEl?.addEventListener('click', (e) => {
        e.stopPropagation();
        const text = ipEl.querySelector('.comp-ip-text')?.textContent ?? '';
        if (!text || text === 'Unknown IP') return;
        navigator.clipboard.writeText(text).then(() => {
          const icon = ipEl.querySelector<SVGElement>('.copy-icon');
          if (!icon) return;
          icon.classList.add('copied');
          setTimeout(() => icon.classList.remove('copied'), 1500);
        });
      });
      menu.appendChild(row);
    });
  }
}

function handleComponentConnStatus(data: nf.telemetry.IComponentConnStatus) {
  let name = "Unknown";
  let type: 'Anchor' | 'Gripper' = 'Anchor';

  if (data.isGripper) {
    type = 'Gripper';
    name = "Arpeggio Gripper";
  } else {
    type = 'Anchor';
    name = `Anchor ${data.anchorNum}`;
  }

  componentStates.set(name, {
    name: name,
    type: type,
    status: data.websocketStatus ?? nf.telemetry.ConnStatus.CONNSTATUS_NOT_DETECTED,
    ip: data.ipAddress ?? "",
    errorMessage: data.errorMessage ?? undefined,
    temp: data.temp ?? undefined,
    motorTorque: data.motorEnabled ?? undefined,
  });

  updateComponentStatusUI();
}

function handleTargetList(data: nf.telemetry.ITargetList) {
  const targets = data.targets ?? [];

  // show the targets in every video feed
  overheadVideofeeds.forEach(feed => {feed.updateList(data)});
  targetListManager.updateList(targets);
}

function showPopup(data: nf.telemetry.IPopup) {
  const overlay = document.getElementById('popup-overlay');
  const msgEl = document.getElementById('popup-message');

  if (overlay && msgEl && data.message) {
    msgEl.textContent = data.message;
    overlay.classList.remove('hidden');
  }
}

function handleNamedPosition(data: nf.telemetry.INamedObjectPosition) {
  if (data.name) {
    if (data.position) {
      // moved the named object
      if (data.name == 'hamper' || data.name == 'toys' || data.name == 'trash' || data.name == 'gamepad') {
        room.setNamedObjectPosition(data.name, data.position);
      }
      // else if (data.name == 'gantry_goal_marker') {

      // }
    } else {
      // hide the named object
    }
  }
}

// --- Target route source/destination pickers ---

const TARGET_ROUTE_SOURCE_OPTIONS = ['All targets', 'User targets', 'Toybox', 'Hamper', 'Trash', 'Gamepad', 'Origin'];
// "All targets" and "User targets" describe groups of targets, not a single named location — only valid as a source.
const TARGET_ROUTE_DESTINATION_OPTIONS = ['Toybox', 'Hamper', 'Trash', 'Gamepad', 'Origin'];

const ROUTE_POINT_BY_LABEL: Record<string, nf.common.RoutePoint> = {
  'All targets': nf.common.RoutePoint.ROUTEPOINT_ALL_TARGETS,
  'User targets': nf.common.RoutePoint.ROUTEPOINT_USER_TARGETS,
  'Toybox': nf.common.RoutePoint.ROUTEPOINT_TOYBOX,
  'Hamper': nf.common.RoutePoint.ROUTEPOINT_HAMPER,
  'Trash': nf.common.RoutePoint.ROUTEPOINT_TRASH,
  'Gamepad': nf.common.RoutePoint.ROUTEPOINT_GAMEPAD,
  'Origin': nf.common.RoutePoint.ROUTEPOINT_ORIGIN,
};

const ROUTE_POINT_LABELS: Record<number, string> = Object.fromEntries(
  Object.entries(ROUTE_POINT_BY_LABEL).map(([label, point]) => [point, label])
);

function initTargetRoutePicker(btnId: string, menuId: string, options: string[], onChange: (value: string) => void) {
  const btn = document.getElementById(btnId);
  const label = btn?.querySelector('.route-select-label');
  const menu = document.getElementById(menuId);
  if (!btn || !label || !menu) return;

  for (const option of options) {
    const item = document.createElement('div');
    item.className = 'menu-item';
    item.textContent = option;
    item.addEventListener('click', (e) => {
      e.stopPropagation();
      menu.classList.remove('show');
      onChange(option);
    });
    menu.appendChild(item);
  }

  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    menu.classList.toggle('show');
  });

  document.addEventListener('click', () => menu.classList.remove('show'));
}

function initTargetRoutePickers() {
  initTargetRoutePicker('route-source-btn', 'route-source-menu', TARGET_ROUTE_SOURCE_OPTIONS, (value) => {
    sendControl([nf.control.ControlItem.create({
      setPoint: { routeSource: ROUTE_POINT_BY_LABEL[value] }
    })]);
  });
  initTargetRoutePicker('route-destination-btn', 'route-destination-menu', TARGET_ROUTE_DESTINATION_OPTIONS, (value) => {
    sendControl([nf.control.ControlItem.create({
      setPoint: { routeDestination: ROUTE_POINT_BY_LABEL[value] }
    })]);
  });
}
initTargetRoutePickers();

// VideoReady.feed_number identifies which stream this is. Note that it is NOT the
// anchor num in stringman pilot. The feeds are:
//   0: gripper camera
//   1: first preferred anchor camera
//   2: second preferred anchor camera
//   3: reprojected floor image (handled by floorProjection below)
//   4: heatmap overlay, transmitted only in debug builds and unused by the UI
//      (falls through to the unknown-feed guard and is ignored)
async function handleVideoReady(data: nf.telemetry.IVideoReady) {
  const feedNumber = data.feedNumber!;

  if (feedNumber === 3) {
    if (isLanMode && data.localUri) {
      floorProjection.connectLocal(data.localUri);
    } else if (data.streamPath) {
      let ticket: string | undefined;
      if (!isSimMode && currentRobotId) {
        try {
          const token = await AuthManager.getAuthToken();
          ticket = await AuthManager.apiGetStreamTicket(currentRobotId, token);
        } catch (e) {
          console.warn("Could not get a stream ticket for floor projection:", e);
        }
      }
      if (currentAccessLevel === 'spectator') {
        floorProjection.connectHLS(data.streamPath, ticket);
      } else {
        floorProjection.connectWebRTC(data.streamPath, ticket);
      }
    }
    return;
  }

  const videoManager = [gripperVideo, firstOverheadVideo, secondOverheadVideo][feedNumber];

  if (!videoManager) {
    console.warn(`handleVideoReady: no video manager for feed number ${feedNumber}, ignoring`);
    return;
  }

  if (isLanMode && data.localUri) {
    // Connect to the video streams at the local UDP address for this camera. example udp:127.0.0.1:1234
    videoManager.connectLocal(data.localUri);
  } else if (data.streamPath) {
    // Connect with WebRTC to the production server at data.streamPath
    let ticket: string | undefined;
    if (!isSimMode && currentRobotId) {
      try {
        const token = await AuthManager.getAuthToken();
        ticket = await AuthManager.apiGetStreamTicket(currentRobotId, token);
      } catch (e) {
        console.warn("Could not get a stream ticket for video auth:", e);
      }
    }
    // Spectators watch the scalable HLS remux instead of a per-viewer WebRTC feed.
    if (currentAccessLevel === 'spectator') {
      videoManager.connectHLS(data.streamPath, ticket);
    } else {
      videoManager.connectWebRTC(data.streamPath, ticket);
    }
  }

  // Assign the feed's anchor num and vitual camera for target overlay math
  if (!data.isGripper && data.anchorNum != null) {
    const anchorIndex = data.anchorNum;
    videoManager.assign(anchorIndex); // anchor num is here
    const corner = corners[anchorIndex];
    if (corner instanceof ArpAnchor) {
      // The anchor's GLTF model (and thus .camera) loads asynchronously and may
      // not be ready yet when this telemetry arrives — wait for it.
      await corner.ready;
      // Re-fetch after the await in case something else replaced this corner in the meantime.
      const loadedCorner = corners[anchorIndex];
      if (loadedCorner instanceof ArpAnchor && loadedCorner.camera) {
        const storedTilt = lastTiltAngles[anchorIndex];
        if (storedTilt != null) {
          applyAnchorCamTilt(anchorIndex, loadedCorner.camera, storedTilt);
        }
        videoManager.setVirtualCamera(loadedCorner.camera);
        attachCameraHelper(anchorIndex, loadedCorner.camera);

        // When the mouse moves on this video feed and a ray intersects the floor
        videoManager.onFloorPoint = (point) => {
          room.setReticule(point)
        };
      }
    }
  }
}

function handleUplinkStatus(data: nf.telemetry.IUplinkStatus) {
  updateOnlineStatus(data.online ?? false);
}

function handleGripSensors(data: nf.telemetry.IGripperSensors) {
  gripper.setSensorValues(data, laserReadings);
  
  if (data.pressure !== undefined && data.pressure !== null) {
    currentGripperPressure = data.pressure;
  }
  
  if (data.targetForce !== undefined && data.targetForce !== null) {
    currentGripperTargetForce = data.targetForce;
  } else {
    currentGripperTargetForce = null;
  }
}

function handleTaskStatus(data: nf.telemetry.ITaskStatus) {
  const sourceLabel = document.querySelector('#route-source-btn .route-select-label');
  if (sourceLabel && data.routeSource !== undefined && data.routeSource !== null) {
    sourceLabel.textContent = ROUTE_POINT_LABELS[data.routeSource] ?? sourceLabel.textContent;
  }

  const destinationLabel = document.querySelector('#route-destination-btn .route-select-label');
  if (destinationLabel && data.routeDestination !== undefined && data.routeDestination !== null) {
    destinationLabel.textContent = ROUTE_POINT_LABELS[data.routeDestination] ?? destinationLabel.textContent;
  }
}

/** True while swing cancellation is currently enabled on the robot. */
export function isSwingCancellationEnabled(): boolean {
  return swingCancellationEnabled;
}

function handleSwingCancellationState(data: nf.telemetry.ISwingCancellationState) {
  swingCancellationEnabled = data.enabled ?? false;
  const btn = document.getElementById('btn-swing-cancel');
  const indicator = document.getElementById('swing-cancel-indicator');

  if (btn && indicator) {
    if (swingCancellationEnabled) {
      btn.classList.add('perspective-button-selected');
      indicator.classList.remove('status-offline');
    } else {
      btn.classList.remove('perspective-button-selected');
      indicator.classList.add('status-offline');
    }
  }
}

function handleTensionRegulationState(data: nf.telemetry.ITensionRegulationState) {
  tensionRegEnabled = data.enabled ?? false;
  const tensionRegEl = document.getElementById('action-toggle-tension-reg');
  if (tensionRegEl) {
    tensionRegEl.textContent = tensionRegEnabled ? 'Disable tension reg' : 'Enable tension reg';
  }
}

function handleAutoTargetingState(data: nf.telemetry.IAutoTargetingState) {
  autoTargetEnabled = data.enabled ?? false;
  const btn = document.getElementById('btn-auto-target');
  btn?.classList.toggle('active', autoTargetEnabled);
}

function updateOnlineStatus(online: boolean) {
  const statusDot = document.getElementById('status-dot-el');
  const statusText = document.getElementById('status-text');
  const runBtn = document.getElementById('run-btn');

  if (statusDot && statusText && runBtn) {
    if (online) {
      statusDot.classList.remove('status-offline');
      
      let text = 'Online';
      if (isLanMode) text = 'Online (Local)';
      if (isSimMode) text = 'Online (Sim)';
      
      statusText.textContent = text;
      runBtn.classList.remove('disabled');
    } else {
      statusDot.classList.add('status-offline');
      statusText.textContent = 'Offline';
      runBtn.classList.add('disabled');

      firstOverheadVideo.setOffline();
      secondOverheadVideo.setOffline();
      gripperVideo.setOffline();
      floorProjection.setOffline();

      componentStates.forEach(comp => {
        comp.status = nf.telemetry.ConnStatus.CONNSTATUS_NOT_DETECTED;
      });
      updateComponentStatusUI();
    }
  }
}

// Function called when mode changes
function onPerspectiveChanged(mode: number) {
  switch (mode) {
  case perspViewport:
    gamepad.setOrbitCenter(camera);
    break;

  case perspPerson:
    if (room.userPers) {
      gamepad.setOrbitCenter(room.userPers);
    }
    break;

  case perspTop:
    if (firstOverheadVideo.virtualCamera) {
      gamepad.setOrbitCenter(firstOverheadVideo.virtualCamera);
    }
    break;

  case perspBottom:
    if (secondOverheadVideo.virtualCamera) {
      gamepad.setOrbitCenter(secondOverheadVideo.virtualCamera);
    }
    break;

  case perspGripper:
    gamepad.setSeatOrbitMode(false);

  }
}

// Helper to handle state change and UI updates
function setPerspective(mode: number) {
    // Update state
  currentPerspective = mode;

    // Update UI classes
  const buttonIds: Record<number, string> = {
    [perspViewport]: 'btn-viewport',
    [perspPerson]: 'btn-person',
    [perspTop]: 'btn-top',
    [perspBottom]: 'btn-bottom',
    [perspGripper]: 'btn-gripper'
  };

    // Remove 'perspective-button-selected' from all buttons
  Object.values(buttonIds).forEach(id => {
    document.getElementById(id)?.classList.remove('perspective-button-selected');
  });

    // Add 'perspective-button-selected' to the active button
  const activeId = buttonIds[mode];
  document.getElementById(activeId)?.classList.add('perspective-button-selected');

  onPerspectiveChanged(mode);
}

// Initialize listeners for perspective button
function initPerspectiveControls() {
  document.getElementById('btn-viewport')?.addEventListener('click', () => setPerspective(perspViewport));
  document.getElementById('btn-person')?.addEventListener('click', () => setPerspective(perspPerson));
  document.getElementById('btn-top')?.addEventListener('click', () => setPerspective(perspTop));
  document.getElementById('btn-bottom')?.addEventListener('click', () => setPerspective(perspBottom));
  document.getElementById('btn-gripper')?.addEventListener('click', () => setPerspective(perspGripper));

  setPerspective(currentPerspective);
}

initPerspectiveControls();

// --- Mobile Shell ---
// Detects mobile (or `?mobile=1`), sets up the tab UI, locks the 3D camera,
// and hosts the simulated joystick. Safe no-op on desktop.
const mobileShell = new MobileShell({
  controls,
  camera,
  renderer,
  setPerspective,
  perspectives: { top: perspTop, bottom: perspBottom, gripper: perspGripper },
});
gamepad.touchProvider = () => mobileShell.getTouchInputState();

// One-time setup to bind the button in the popup window
function initPopup() {
  const overlay = document.getElementById('popup-overlay');
  const btn = document.getElementById('popup-ok');

  if (overlay && btn) {
    btn.addEventListener('click', () => {
      overlay.classList.add('hidden');
    });
  }
}

initPopup();

// Send a list of ControlItems immediately
function sendControl(items: Array<nf.control.ControlItem>) {
  // Spectators are view-only: never emit control messages. (The backend also
  // drops them, but this avoids pointless traffic and keeps the UI honest.)
  if (!canSendControl()) return;

  // Determine robot ID context
  let rId = currentRobotId ?? "unknown";
  if (isLanMode) rId = "local";
  if (isSimMode) rId = "simulated";

  const batchData = nf.control.ControlBatchUpdate.create({
    robotId: rId,
    updates: items
  });

  // protobufjs method of serialization
  const writer = nf.control.ControlBatchUpdate.encode(batchData);
  const serializedBinaryData = writer.finish();
  if (socket && socket.readyState === WebSocket.OPEN) {
    // agents ignore the error on the following line.
    socket.send(serializedBinaryData);
  }
}

// Send a single command to the robot.
function simpleCommand(cmdEnum: nf.control.Command) {
  sendControl([nf.control.ControlItem.create({
    command: {name: cmdEnum}
  })]);
}

// send a command that concerns a single component
function sendSingleComponentAction(type: string, index: number, actionEnum: nf.control.ComponentAction) {
  const isGripper = (type === 'Gripper');
  const action = nf.control.ControlItem.create({
    singleComponentAction: {
      isGripper: isGripper,
      anchorNum: isGripper ? undefined : index,
      action: actionEnum
    }
  });
  sendControl([action]);
}

overheadVideofeeds.forEach(feed => {feed.sendFn = sendControl});
targetListManager.sendFn = sendControl;

// --- Run menu ---
function initRunMenu() {
  const runBtn = document.getElementById('run-btn');
  const runMenu = document.getElementById('run-menu');
  const maintMenu = document.getElementById('maintenance-menu');
  const stopBtn = document.getElementById('stop-btn');

    // Toggle Main Menu
  if (runBtn && runMenu) {
    runBtn.addEventListener('click', (e) => {
      e.stopPropagation();
            // Close maint menu if open, toggle main menu
      maintMenu?.classList.remove('show');
      runMenu.classList.toggle('show');

      // Pick and drop / auto grasp require every anchor and gripper to be connected.
      const actionsAvailable = isFullyConnected() && isFullyCalibrated();
      document.getElementById('action-pick-drop')?.classList.toggle('disabled', !actionsAvailable);
      document.getElementById('action-grasp')?.classList.toggle('disabled', !actionsAvailable);
    });
  }

    // Toggle Maintenance Menu
  const maintAction = document.getElementById('action-maintenance');
  const maintBack = document.getElementById('action-maint-back');

  if (maintAction && maintMenu && runMenu) {
    maintAction.addEventListener('click', (e) => {
      e.stopPropagation();
      runMenu.classList.remove('show');
      maintMenu.classList.add('show');
    });
  }

  if (maintBack && maintMenu && runMenu) {
    maintBack.addEventListener('click', (e) => {
      e.stopPropagation();
      maintMenu.classList.remove('show');
      runMenu.classList.add('show');
    });
  }

    // Prevent input click from closing menu
  const debugInput = document.getElementById('debug-input');
  if (debugInput) {
    debugInput.addEventListener('click', (e) => e.stopPropagation());
  }

    // Debug Send Action
  const debugSendBtn = document.getElementById('action-debug-send');
  if (debugSendBtn && debugInput) {
    debugSendBtn.addEventListener('click', () => {
      const val = (debugInput as HTMLInputElement).value;
      if (val) {
        console.log("Sending debug cmd:", val);
        sendControl([nf.control.ControlItem.create({
          debug: {action: val}
        })]);
        maintMenu?.classList.remove('show');
      }
    });
  }

    // Close menu when clicking outside
  document.addEventListener('click', () => {
    if (runMenu && runMenu.classList.contains('show')) {
      runMenu.classList.remove('show');
    }
    if (maintMenu && maintMenu.classList.contains('show')) {
      maintMenu.classList.remove('show');
    }
  });

    // Helper to bind menu items to simpleCommand
  const bindCommand = (elementId: string, cmdEnum: nf.control.Command) => {
    const el = document.getElementById(elementId);
    if (el) {
      el.addEventListener('click', () => {
        if (!el.classList.contains('disabled')) {
          simpleCommand(cmdEnum);
                    // close menu after selection
          runMenu?.classList.remove('show');
          maintMenu?.classList.remove('show');
        }
      });
    }
  };

    // Bind all menu actions
  const Command = nf.control.Command;

  bindCommand('action-pick-drop',       Command.COMMAND_PICK_AND_DROP);
  // bindCommand('action-tension',        Command.COMMAND_TIGHTEN_LINES);
  document.getElementById('action-full-cal')?.addEventListener('click', () => {
    if (document.getElementById('action-full-cal')?.classList.contains('disabled')) return;
    runMenu?.classList.remove('show');
    maintMenu?.classList.remove('show');
    openFullCalOverlay();
  });
  bindCommand('action-half-cal',       Command.COMMAND_HALF_CAL);
  bindCommand('action-grasp',          Command.COMMAND_GRASP);
  bindCommand('action-dataset',        Command.COMMAND_SUBMIT_TARGETS_TO_DATASET);
  bindCommand('action-collect-images', Command.COMMAND_COLLECT_GRIPPER_IMAGES);
  bindCommand('action-update-firmware', Command.COMMAND_UPDATE_FIRMWARE);
  bindCommand('action-record-park',    Command.COMMAND_RECORD_PARK);
  // Park / UnPark temporarily removed from the menu — feature is unstable and unrecommended.
  bindCommand('action-disable-torque', Command.COMMAND_DISABLE_TORQUE);
  bindCommand('action-enable-torque', Command.COMMAND_ENABLE_TORQUE);

  // Tension regulation toggle. The label reflects the action the next press will take;
  // actual state comes from tension_regulation_state telemetry (see handleTensionRegulationState).
  const tensionRegEl = document.getElementById('action-toggle-tension-reg');
  if (tensionRegEl) {
    tensionRegEl.addEventListener('click', () => {
      if (tensionRegEl.classList.contains('disabled')) return;
      simpleCommand(tensionRegEnabled ? Command.COMMAND_DISABLE_TENSION_REG : Command.COMMAND_ENABLE_TENSION_REG);
      runMenu?.classList.remove('show');
      maintMenu?.classList.remove('show');
    });
  }

  document.getElementById('action-get-ticket')?.addEventListener('click', () => {
    maintMenu?.classList.remove('show');
    handleGetTicket();
  });


  // Log panel toggle (from the Calibration & Maintenance menu)
  document.getElementById('action-toggle-logs')?.addEventListener('click', () => {
    maintMenu?.classList.remove('show');
    toggleLogPanel();
  });
  document.getElementById('log-panel-close')?.addEventListener('click', () => {
    hideLogPanel();
  });

  // Bind bind action (connecting your robot to your account so you can use cloud relay)
  const bindAction = document.getElementById('action-bind');
  if (bindAction) {
    bindAction.addEventListener('click', () => {
      handleBindAction();
      runMenu?.classList.remove('show');
    });
  }

  // stop
  if (stopBtn) {
    stopBtn.addEventListener('click', () => {
      console.log("Stop current task");
      simpleCommand(Command.COMMAND_STOP_ALL);
    });
  }
}

// Initialize run menu listeners
initRunMenu();

// --- Header Menu ---
function initHeader() {
  document.getElementById('btn-header-back')?.addEventListener('click', () => {
    // Disconnect and stop reconnection loops
    disconnect();
    // Show landing layer
    document.getElementById('landing-layer')?.classList.remove('hidden');
    // Ensure we are on the main options screen, not the robot list or bind screen
    document.getElementById('landing-options')?.classList.remove('hidden');
    document.getElementById('robot-list-panel')?.classList.add('hidden');
    document.getElementById('bind-robot-panel')?.classList.add('hidden');
  });

  // robot list back button
  document.getElementById('btn-robot-list-back')?.addEventListener('click', () => {
    document.getElementById('robot-list-panel')?.classList.add('hidden');
    document.getElementById('landing-layer')?.classList.remove('hidden');
    document.getElementById('landing-options')?.classList.remove('hidden');
  });

  // sign-in overlay: close on X button or background click
  document.getElementById('btn-signin-back')?.addEventListener('click', () => {
    AuthManager.hideSignInUI();
  });
  document.getElementById('signin-overlay')?.addEventListener('click', (e) => {
    if (e.target === e.currentTarget) AuthManager.hideSignInUI();
  });
}
initHeader();

// --- Controls Panel ---
function initControlsPanel() {
  const btnControls = document.getElementById('btn-header-controls');
  const overlay = document.getElementById('controls-overlay');
  const closeBtn = document.getElementById('close-controls-panel');
  const catcher = document.getElementById('controls-bg-catcher');

  btnControls?.addEventListener('click', () => {
    overlay?.classList.remove('hidden');
  });

  const closePanel = () => overlay?.classList.add('hidden');
  closeBtn?.addEventListener('click', closePanel);
  catcher?.addEventListener('click', closePanel);
}
initControlsPanel();

async function handleGetTicket() {
  if (!detectedRobotId) {
    showPopup({ message: "No robot ID detected. Connect to a robot first." });
    return;
  }
  try {
    const token = await AuthManager.getAuthToken();
    const ticket = await AuthManager.apiGetStreamTicket(detectedRobotId, token);

    const overlay = document.getElementById('popup-overlay');
    const msgEl = document.getElementById('popup-message');
    if (!overlay || !msgEl) return;

    msgEl.style.whiteSpace = 'pre-wrap';
    msgEl.style.wordBreak = 'break-all';
    msgEl.style.fontFamily = 'monospace';
    msgEl.style.fontSize = '0.8rem';

    try {
      await navigator.clipboard.writeText(ticket);
      msgEl.textContent = `Stream ticket (copied to clipboard):\n\n${ticket}\n\nMust be used with stringman_lerobot.py within 60 seconds.`;
    } catch {
      msgEl.textContent = `Stream ticket:\n\n${ticket}`;
    }
    overlay.classList.remove('hidden');
  } catch (e) {
    showPopup({ message: "Failed to get ticket. Are you logged in?" });
  }
}

// --- Share Access Dialog ---

async function handleShareAction() {
    const overlay = document.getElementById('share-overlay');
    overlay?.classList.remove('hidden');
    await openShareDialog();
}

async function openShareDialog() {
    const container = document.getElementById('share-list-container');
    if (!container || !detectedRobotId) return;
    container.innerHTML = '<div style="padding:15px; color:#888;">Loading...</div>';

    const token = await AuthManager.getAuthToken();
    const guests = await AuthManager.apiListAuthorized(detectedRobotId, token);

    container.innerHTML = '';
    if (guests.length === 0) {
        container.innerHTML = '<div style="padding:15px; color:#666; font-size:0.85rem;">Not shared with anyone yet.</div>';
        return;
    }

    const LEVEL_LABELS: Record<string, string> = {
        full: 'Full', limited_driver: 'Limited', spectator: 'Spectator',
    };

    guests.forEach(g => {
        const row = document.createElement('div');
        row.className = 'share-user-row';
        const levelLabel = LEVEL_LABELS[g.access_level] ?? g.access_level;
        row.innerHTML = `
            <span class="share-email">${g.email}</span>
            <span class="share-level" style="color:#fff; opacity:0.7; font-size:0.8rem; margin:0 8px;">${levelLabel}</span>
            <button class="btn-revoke" data-email="${g.email}">Revoke</button>
        `;
        const btn = row.querySelector('.btn-revoke') as HTMLButtonElement;
        btn.onclick = async () => {
            if (btn.classList.contains('confirming')) {
                btn.disabled = true;
                btn.textContent = "...";
                await executeRevoke(g.email);
                openShareDialog(); // refresh
            } else {
                btn.textContent = "confirm?";
                btn.classList.add('confirming');
                // Reset others
                container.querySelectorAll('.btn-revoke').forEach(other => {
                    if (other !== btn) { other.textContent = "Revoke"; other.classList.remove('confirming'); }
                });
            }
        };
        container.appendChild(row);
    });
}

async function executeRevoke(email: string) {
    if (!detectedRobotId) return;
    const token = await AuthManager.getAuthToken();
    await fetch(`/share/${detectedRobotId}/${encodeURIComponent(email)}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
    });
}

// --- LeRobot Recording Panel ---

function sendEpisodeCommand(commandVal: nf.common.EpCommand) {
  sendControl([nf.control.ControlItem.create({
    episodeControl: { command: commandVal }
  })]);
}

function sendPrompt(text: string) {
  sendControl([nf.control.ControlItem.create({
    episodeControl: { prompt: text }
  })]);
  const display = document.getElementById('lerobot-last-prompt');
  if (display) display.textContent = text ? `"${text}"` : '';
}

async function triggerSetPrompt() {
  if (!isLeRobotSessionActive) return;
  const btn = document.getElementById('btn-lerobot-set-prompt') as HTMLButtonElement | null;
  if (btn) { btn.textContent = 'Listening...'; btn.disabled = true; }
  try {
    const transcript = await Listen();
    const input = document.getElementById('lerobot-prompt-input') as HTMLInputElement | null;
    if (input) input.value = transcript;
    sendPrompt(transcript);
  } catch (e) {
    console.warn('Speech recognition error:', e);
  } finally {
    if (btn) { btn.textContent = '🎤 Set Prompt'; btn.disabled = false; }
  }
}

// Renders the LeRobot panel (header button, inactive "start" view, active session view)
// from the module-level state variables below. Call this after any of them change.
//
// - isLeRobotSessionActive: master switch between the inactive panel (start buttons)
//   and the active panel (session info + action buttons). Derived in handleEpisodeControl
//   from the latest status: false for NA/null and the terminal REC_ALL_COMPLETE/
//   EVAL_ALL_COMPLETE/ERROR states, true otherwise.
// - leRobotState (LerobotStatus enum): the specific state within either panel - which
//   action buttons render (Start Episode / Complete+Abandon / Stop Episode), the header
//   icon, and the REC_ALL_COMPLETE "successful finish" special case that overrides the
//   normal active/inactive split.
// - isLeRobotStarting: true only during the brief window after clicking a Start button
//   and before the first status update arrives (handleLeRobotStart sets it true;
//   handleEpisodeControl clears it on the next status). Shows the pending spinner and
//   disables all three start buttons; they're re-enabled once this clears.
// - sentFinalizeCommand: true after clicking End Session, before the terminal status
//   arrives. Shows the "FINALIZING DATASET..." spinner in place of the action buttons,
//   except when leRobotState is ERROR (nothing to wait for - see handleLeRobotFinalize).
// - lerobotError: latest error string from status.error, if any. Rendered in the active
//   panel's error box when a session is active, or the inactive panel's start-error box
//   otherwise.
// - numEpisodesRecorded, datasetEpCount, hfRepoId, policyRepoId, episodesUntilCheckpoint:
//   session/progress display fields, all populated from the status payload.
function updateLeRobotUI() {
  const headerBtn = document.getElementById('btn-header-lerobot');
  const panelInactive = document.getElementById('lerobot-panel-inactive');
  const panelActive = document.getElementById('lerobot-panel-active');
  const inactiveContent = document.getElementById('lerobot-inactive-content');
  const pendingOverlay = document.getElementById('lerobot-pending-overlay');

  const startRecBtnLinked = document.getElementById('btn-lerobot-start-rec-linked') as HTMLButtonElement;
  const startRecBtnLan = document.getElementById('btn-lerobot-start-rec-lan') as HTMLButtonElement;
  const startEvalBtn = document.getElementById('btn-lerobot-start-eval') as HTMLButtonElement;

  const sessionEpsEl = document.getElementById('lerobot-session-eps');
  const totalEpsEl = document.getElementById('lerobot-total-eps');
  const repoEl = document.getElementById('lerobot-repo-id');
  const repoLabel = document.getElementById('lerobot-repo-label');
  const errorBox = document.getElementById('lerobot-error-box');
  const titleEl = document.getElementById('lerobot-title');
  const actionButtons = document.getElementById('lerobot-action-buttons');
  const completionLink = document.getElementById('lerobot-completion-link');
  const hfBtn = document.getElementById('hf-vis-btn') as HTMLAnchorElement;

  if (headerBtn) {
    if (isSimMode || !canUseLeRobot()) {
      headerBtn.classList.add('hidden');
    } else {
      headerBtn.classList.remove('hidden');
      if (isLeRobotSessionActive) {
        let icon = '';
        const iconStyle = 'display: inline-block; margin-right: 6px; vertical-align: middle; color: #f44336;';
        if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_REC_READY || leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_EVAL_IDLE) {
          icon = `<span style="${iconStyle} font-size: 1.2em;">○</span>`;
        } else if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_RECORDING || leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_EVAL_ACTIVE) {
          icon = `<span style="${iconStyle} font-size: 1.2em;">●</span>`;
        } else if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_REC_PROCESSING) {
          icon = `<span style="${iconStyle} font-weight: bold; letter-spacing: 1px;">⸭</span>`;
        }
        headerBtn.innerHTML = `${icon} Session Connected`;
      } else {
        headerBtn.textContent = "Record";
      }
    }
  }

  if (panelInactive && panelActive && actionButtons) {
    // Special case: Successful Session Finish
    if (!isLeRobotSessionActive && leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_REC_ALL_COMPLETE) {
      panelInactive.classList.add('hidden');
      panelActive.classList.remove('hidden');
      completionLink?.classList.remove('hidden');
      if (hfBtn) {
        const encodedPath = encodeURIComponent('/' + hfRepoId.replace(/^\//, ''));
        hfBtn.href = `https://huggingface.co/spaces/lerobot/visualize_dataset?path=${encodedPath}`;
      }
      actionButtons.innerHTML = '';
      document.getElementById('btn-lerobot-finalize')?.classList.add('hidden');
      document.getElementById('lerobot-prompt-section')?.classList.add('hidden');
      return;
    }

    if (isLeRobotSessionActive) {
      panelInactive.classList.add('hidden');
      panelActive.classList.remove('hidden');
      completionLink?.classList.add('hidden');

      const isEval = (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_EVAL_IDLE || leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_EVAL_ACTIVE);
      if (titleEl) titleEl.textContent = "LeRobot Session";
      if (repoLabel) repoLabel.textContent = isEval ? "Policy" : "Dataset";
      if (repoEl) repoEl.textContent = isEval ? policyRepoId : hfRepoId;

      if (sessionEpsEl) sessionEpsEl.textContent = numEpisodesRecorded.toString();
      if (totalEpsEl) totalEpsEl.textContent = (datasetEpCount + numEpisodesRecorded).toString();

      const checkpointLine = document.getElementById('lerobot-checkpoint-line');
      const checkpointEl = document.getElementById('lerobot-eps-checkpoint');
      if (!isEval && checkpointLine && checkpointEl) {
        if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_REC_CHECKPOINT) {
          checkpointLine.classList.remove('hidden');
          checkpointEl.textContent = 'Uploading...';
        } else if (episodesUntilCheckpoint !== null) {
          checkpointLine.classList.remove('hidden');
          checkpointEl.textContent = episodesUntilCheckpoint.toString();
        } else {
          checkpointLine.classList.add('hidden');
        }
      } else if (checkpointLine) {
        checkpointLine.classList.add('hidden');
      }

      if (lerobotError && errorBox) {
        errorBox.textContent = lerobotError;
        errorBox.classList.remove('hidden');
      } else if (errorBox) { errorBox.classList.add('hidden'); }

      actionButtons.innerHTML = '';

            // Handle Spinner state during Finalization within the active panel
      if (sentFinalizeCommand && leRobotState !== nf.common.LerobotStatus.LEROBOTSTATUS_ERROR) {
        actionButtons.innerHTML = `
                    <div class="pending-container">
                        <div class="spinner"></div>
                        <div class="starting-text">FINALIZING DATASET...</div>
                    </div>
        `;
        document.getElementById('btn-lerobot-finalize')?.classList.add('hidden');
        document.getElementById('lerobot-prompt-section')?.classList.add('hidden');
      } else {
        document.getElementById('btn-lerobot-finalize')?.classList.remove('hidden');
        document.getElementById('lerobot-prompt-section')?.classList.remove('hidden');

        if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_REC_READY || leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_EVAL_IDLE) {
          const btn = document.createElement('button');
          btn.className = 'run-button btn-green';
          btn.style.width = '100%';
          btn.style.justifyContent = 'center';
          btn.textContent = 'Start Episode';
          btn.onclick = () => sendEpisodeCommand(nf.common.EpCommand.EPCOMMAND_EVAL_START);
          actionButtons.appendChild(btn);
        } 
        else if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_RECORDING) {
          const wrapper = document.createElement('div');
          wrapper.className = 'side-by-side-container';

          const btnComp = document.createElement('button');
          btnComp.className = 'run-button btn-green';
          btnComp.style.flex = '1';
          btnComp.style.justifyContent = 'center';
          btnComp.textContent = 'Complete Episode';
          btnComp.onclick = () => sendEpisodeCommand(nf.common.EpCommand.EPCOMMAND_EVAL_STOP);

          const btnAban = document.createElement('button');
          btnAban.className = 'run-button btn-red';
          btnAban.style.flex = '1';
          btnAban.style.justifyContent = 'center';
          btnAban.textContent = 'Abandon Episode';
          btnAban.onclick = () => sendEpisodeCommand(nf.common.EpCommand.EPCOMMAND_ABANDON);

          wrapper.appendChild(btnComp);
          wrapper.appendChild(btnAban);
          actionButtons.appendChild(wrapper);
        }
        else if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_EVAL_ACTIVE) {
          const btn = document.createElement('button');
          btn.className = 'run-button btn-green';
          btn.style.width = '100%';
          btn.style.justifyContent = 'center';
          btn.textContent = 'Stop Episode';
          btn.onclick = () => sendEpisodeCommand(nf.common.EpCommand.EPCOMMAND_EVAL_STOP);
          actionButtons.appendChild(btn);
        }
      }

    } else {
      panelInactive.classList.remove('hidden');
      panelActive.classList.add('hidden');

      if (isLeRobotStarting) {
        inactiveContent?.classList.add('hidden');
        pendingOverlay?.classList.remove('hidden');
        if (startRecBtnLinked) startRecBtnLinked.disabled = true;
        if (startRecBtnLan) startRecBtnLan.disabled = true;
        if (startEvalBtn) startEvalBtn.disabled = true;
      } else {
        pendingOverlay?.classList.add('hidden');
        const unavailableEl = document.getElementById('lerobot-unavailable');
        if (!isFullyConnected()) {
          inactiveContent?.classList.add('hidden');
          unavailableEl?.classList.remove('hidden');
        } else {
          unavailableEl?.classList.add('hidden');
          inactiveContent?.classList.remove('hidden');
          if (startRecBtnLinked) startRecBtnLinked.disabled = false;
          if (startRecBtnLan) startRecBtnLan.disabled = false;
          if (startEvalBtn) startEvalBtn.disabled = false;
          updateSessionLocationUI();

          const startErrorEl = document.getElementById('lerobot-start-error');
          if (startErrorEl) {
            if (lerobotError) {
              startErrorEl.textContent = lerobotError;
              startErrorEl.classList.remove('hidden');
            } else {
              startErrorEl.classList.add('hidden');
            }
          }
        }
      }
    }
  }
}

function handleLeRobotStart(type: 'recording' | 'eval', repoId: string) {
  isLeRobotStarting = true;
  lerobotError = null;
  sentFinalizeCommand = false;
  episodesUntilCheckpoint = null;

  if (type === 'recording') {
    sendControl([nf.control.ControlItem.create({
      manageLerobotSession: { action: nf.control.LerobotSessionAction.LEROBOTSESSIONACTION_START_RECORD, repoId: repoId }
    })]);
  } else {
    sendControl([nf.control.ControlItem.create({
      manageLerobotSession: { action: nf.control.LerobotSessionAction.LEROBOTSESSIONACTION_START_EVAL, repoId: repoId }
    })]);
  }
  updateLeRobotUI();
}

function handleLeRobotFinalize() {
    // if there is an error status let the button fix the UI state, otherwise send a message
  if (leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_ERROR) {
    isLeRobotSessionActive = false;
    updateLeRobotUI();
  } else {
    sentFinalizeCommand = true;
    sendEpisodeCommand(nf.common.EpCommand.EPCOMMAND_END_RECORDING);
    updateLeRobotUI();
  }
}

async function refreshHfStatus() {
  const lanSection = document.getElementById('lerobot-rec-lan');
  const linkedInputs = document.getElementById('lerobot-rec-linked-inputs');
  const locationLan = document.getElementById('lerobot-location-lan');

  lanSection?.classList.add('hidden');
  linkedInputs?.classList.add('hidden');
  locationLan?.classList.add('hidden');

  if (isLanMode) {
    lanSection?.classList.remove('hidden');
    locationLan?.classList.remove('hidden');
    return;
  }
  linkedInputs?.classList.remove('hidden');

  try {
    const token = await AuthManager.getAuthToken();
    const res = await fetch('/huggingface/status', {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    const data = await res.json();
    const usernameEl = document.getElementById('lerobot-hf-username');
    if (usernameEl && data.hf_username) usernameEl.textContent = data.hf_username;
  } catch (e) {
    console.warn('Could not fetch HF status:', e);
  }
  updateSessionLocationUI();
}

function updateSessionLocationUI() {
  const usernameEl = document.getElementById('lerobot-hf-username');
  const slashEl = document.getElementById('lerobot-dataset-slash');
  const nameInput = document.getElementById('lerobot-input-dataset-name') as HTMLInputElement | null;

  // Recording always runs as a subprocess of stringman-headless, so the dataset
  // is given as a full "owner/name" repo id rather than split into username + name.
  if (usernameEl) usernameEl.style.display = 'none';
  if (slashEl) slashEl.style.display = 'none';
  if (nameInput) nameInput.placeholder = 'username/my-dataset';
}

function initLeRobotPanel() {
  const headerBtn = document.getElementById('btn-header-lerobot');
  const overlay = document.getElementById('lerobot-overlay');
  const closeBtn = document.getElementById('close-lerobot-panel');
  const catcher = document.getElementById('lerobot-bg-catcher');
  const startEvalBtn = document.getElementById('btn-lerobot-start-eval');
  const finalizeBtn = document.getElementById('btn-lerobot-finalize');

  headerBtn?.addEventListener('click', () => {
    if (isSimMode) return;
    overlay?.classList.remove('hidden');
    updateLeRobotUI();
    if (!isLeRobotSessionActive) {
      refreshHfStatus();
    }
  });

  const closePanel = () => {
    overlay?.classList.add('hidden');
    if (!isLeRobotSessionActive && leRobotState === nf.common.LerobotStatus.LEROBOTSTATUS_REC_ALL_COMPLETE) {
      leRobotState = nf.common.LerobotStatus.LEROBOTSTATUS_NA;
    }
  }
  closeBtn?.addEventListener('click', closePanel);
  catcher?.addEventListener('click', closePanel);

  document.getElementById('btn-lerobot-start-rec-linked')?.addEventListener('click', () => {
    const nameInput = document.getElementById('lerobot-input-dataset-name') as HTMLInputElement;
    handleLeRobotStart('recording', nameInput?.value ?? '');
  });

  document.getElementById('btn-lerobot-start-rec-lan')?.addEventListener('click', () => {
    const input = document.getElementById('lerobot-input-dataset') as HTMLInputElement;
    handleLeRobotStart('recording', input.value);
  });

  startEvalBtn?.addEventListener('click', () => {
    const input = document.getElementById('lerobot-input-policy') as HTMLInputElement;
    handleLeRobotStart('eval', input.value);
  });

  finalizeBtn?.addEventListener('click', () => {
    handleLeRobotFinalize();
  });

  document.getElementById('btn-lerobot-set-prompt')?.addEventListener('click', () => {
    triggerSetPrompt();
  });

  const promptInput = document.getElementById('lerobot-prompt-input') as HTMLInputElement | null;
  promptInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendPrompt(promptInput.value);
  });
  document.getElementById('btn-lerobot-send-prompt')?.addEventListener('click', () => {
    if (promptInput) sendPrompt(promptInput.value);
  });

  updateLeRobotUI();
}
initLeRobotPanel();

function openFullCalOverlay() {
  const overlay = document.getElementById('fullcal-overlay');
  const content = document.getElementById('fullcal-content');
  const unavailable = document.getElementById('fullcal-unavailable');
  if (!overlay) return;
  if (!isFullyConnected()) {
    content?.classList.add('hidden');
    unavailable?.classList.remove('hidden');
  } else {
    unavailable?.classList.add('hidden');
    content?.classList.remove('hidden');
  }
  if (lastTiltAngles[0] != null) {
    (document.getElementById('fullcal-angle-0') as HTMLInputElement).value = lastTiltAngles[0].toString();
  }
  if (lastTiltAngles[1] != null) {
    (document.getElementById('fullcal-angle-1') as HTMLInputElement).value = lastTiltAngles[1].toString();
  }
  overlay.classList.remove('hidden');
}

function initFullCalPanel() {
  const close = () => document.getElementById('fullcal-overlay')?.classList.add('hidden');
  document.getElementById('close-fullcal-panel')?.addEventListener('click', close);
  document.getElementById('fullcal-bg-catcher')?.addEventListener('click', close);

  document.getElementById('btn-fullcal-start')?.addEventListener('click', () => {
    const angle0 = parseFloat((document.getElementById('fullcal-angle-0') as HTMLInputElement)?.value ?? String(DEFAULT_CAM_TILT));
    const angle1 = parseFloat((document.getElementById('fullcal-angle-1') as HTMLInputElement)?.value ?? String(DEFAULT_CAM_TILT));

    sendControl([
      nf.control.ControlItem.create({
        singleComponentAction: {
          isGripper: false, anchorNum: 0,
          action: nf.control.ComponentAction.COMPONENTACTION_SET_CAM_ANGLE,
          camAngle: isNaN(angle0) ? DEFAULT_CAM_TILT : angle0,
        }
      }),
      nf.control.ControlItem.create({
        singleComponentAction: {
          isGripper: false, anchorNum: 1,
          action: nf.control.ComponentAction.COMPONENTACTION_SET_CAM_ANGLE,
          camAngle: isNaN(angle1) ? DEFAULT_CAM_TILT : angle1,
        }
      }),
    ]);

    simpleCommand(nf.control.Command.COMMAND_FULL_CAL);
    close();
  });
}
initFullCalPanel();

function initSharePanel() {
    document.getElementById('close-share-panel')?.addEventListener('click', () => document.getElementById('share-overlay')?.classList.add('hidden'));
    document.getElementById('share-bg-catcher')?.addEventListener('click', () => document.getElementById('share-overlay')?.classList.add('hidden'));
    document.getElementById('btn-share-submit')?.addEventListener('click', async () => {
        const input = document.getElementById('share-input-email') as HTMLInputElement;
        const levelSelect = document.getElementById('share-input-level') as HTMLSelectElement;
        const email = input.value.trim();
        if (!email || !detectedRobotId) return;
        const level = levelSelect?.value || 'full';
        const btn = document.getElementById('btn-share-submit') as HTMLButtonElement;
        btn.disabled = true; btn.textContent = "Sharing...";
        try {
            const token = await AuthManager.getAuthToken();
            await AuthManager.apiShareRobot(detectedRobotId, email, level, token);
            input.value = ''; openShareDialog();
        } finally { btn.disabled = false; btn.textContent = "Share Access"; }
    });
}
initSharePanel();

// --- Video Feed Expand ---
let expandedFeedId: string | null = null;
const ALL_FEED_IDS = ['firstOverhead', 'secondOverhead', 'gripper'];

function expandFeed(feedId: string, perspective: number) {
  if (expandedFeedId) collapseFeed();
  const panelRight = document.getElementById('panel-right');
  const feed = document.getElementById(feedId);
  if (!panelRight || !feed) return;

  for (const id of ALL_FEED_IDS) {
    if (id !== feedId) document.getElementById(id)?.classList.add('hidden');
  }

  panelRight.classList.add('panel-right-expanded');
  const rect = panelRight.getBoundingClientRect(); // measured after grid reflow
  feed.style.position = 'fixed';
  feed.style.left = `${rect.left}px`;
  feed.style.top = `${rect.top}px`;
  feed.style.width = `${rect.width}px`;
  feed.style.height = `${rect.height}px`;
  feed.style.zIndex = '50';
  feed.style.aspectRatio = 'unset';
  feed.style.borderRadius = '12px';

  panelRight.style.overflowY = 'hidden';
  feed.querySelector<HTMLElement>('.feed-expand-btn')?.classList.add('hidden');
  feed.querySelector<HTMLElement>('.feed-collapse-btn')?.classList.remove('hidden');

  expandedFeedId = feedId;
  setPerspective(perspective);
}

function collapseFeed() {
  if (!expandedFeedId) return;
  const feed = document.getElementById(expandedFeedId);
  const panelRight = document.getElementById('panel-right');

  for (const id of ALL_FEED_IDS) {
    document.getElementById(id)?.classList.remove('hidden');
  }

  if (feed) {
    feed.style.position = '';
    feed.style.left = '';
    feed.style.top = '';
    feed.style.width = '';
    feed.style.height = '';
    feed.style.zIndex = '';
    feed.style.aspectRatio = '';
    feed.style.borderRadius = '';
    feed.querySelector<HTMLElement>('.feed-expand-btn')?.classList.remove('hidden');
    feed.querySelector<HTMLElement>('.feed-collapse-btn')?.classList.add('hidden');
  }
  if (panelRight) {
    panelRight.style.overflowY = '';
    panelRight.classList.remove('panel-right-expanded');
  }
  expandedFeedId = null;
}

function initFeedExpand() {
  const feeds: Array<{ id: string; perspective: number }> = [
    { id: 'firstOverhead',  perspective: perspTop },
    { id: 'secondOverhead', perspective: perspBottom },
    { id: 'gripper',        perspective: perspGripper },
  ];
  for (const { id, perspective } of feeds) {
    const feed = document.getElementById(id);
    feed?.querySelector('.feed-expand-btn')?.addEventListener('click', () => expandFeed(id, perspective));
    feed?.querySelector('.feed-collapse-btn')?.addEventListener('click', collapseFeed);
  }
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && expandedFeedId) collapseFeed();
  });
}
initFeedExpand();

// --- Component Status Menu ---
function initComponentMenu() {
  const compBtn = document.getElementById('component-status-btn');
  const compMenu = document.getElementById('component-menu');

  if (compBtn && compMenu) {
    compBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      compMenu.classList.toggle('show');
    });
  }

  // Close menu when clicking outside (shares logic with run menu if needed, or specific listener)
  document.addEventListener('click', () => {
    if (compMenu && compMenu.classList.contains('show')) {
      compMenu.classList.remove('show');
    }
  });
}

initComponentMenu();

function toggleSwingCancellation() {
  if (!isFullyConnected() || !isFullyCalibrated()) return;
  sendControl([nf.control.ControlItem.create({
    setSwingCancellation: { enabled: !swingCancellationEnabled, present: '.' }
  })]);
}

/** Swing cancellation requires every anchor/gripper connected and full calibration to be meaningful. */
function updateSwingToggleAvailability() {
  const available = isFullyConnected() && isFullyCalibrated();
  const btn = document.getElementById('btn-swing-cancel');
  btn?.classList.toggle('disabled', !available);
  if (btn) {
    btn.title = available
      ? 'Actively cancel gripper swing. (left stick click)'
      : 'Swing cancellation is only available after calibration';
  }
}

function toggleAutoTarget() {
  sendControl([nf.control.ControlItem.create({
    setTargetModel: {
      action: autoTargetEnabled
        ? nf.control.TargetModelAction.TARGET_MODEL_DISABLE
        : nf.control.TargetModelAction.TARGET_MODEL_ENABLE_DEFAULT
    }
  })]);
}
document.getElementById('btn-auto-target')?.addEventListener('click', toggleAutoTarget);

// --- Swing Control Menu ---
function initSwingControl() {
  const btn = document.getElementById('btn-swing-cancel');
  btn?.addEventListener('click', toggleSwingCancellation);
  gamepad.toggleSwingC = toggleSwingCancellation;
  gamepad.onSetPrompt = triggerSetPrompt;

  const slider = document.getElementById('swing-latency-slider') as HTMLInputElement | null;
  const valueDisplay = document.getElementById('swing-latency-value');
  slider?.addEventListener('input', () => {
    const val = parseFloat(slider.value).toFixed(2);
    if (valueDisplay) valueDisplay.textContent = val + 's';
    sendControl([nf.control.ControlItem.create({
      debug: { action: `swinglatency ${val}` }
    })]);
  });

  updateSwingToggleAvailability();
}
initSwingControl();

// --- Component Interaction & Details Panel ---

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let currentHoverType: string | null = null;
let currentHoverIndex: number = -1;
let activeComponentData: { type: string, index: number, name: string } | null = null;
let lastTiltAngles: number[] = [];
// Tracks the tilt angle (degrees) currently applied to each anchor's camera so deltas are correct.
const appliedCamTilt: (number | null)[] = [null, null];

function applyAnchorCamTilt(anchorIndex: number, camera: THREE.PerspectiveCamera, tiltDeg: number) {
  const prev = appliedCamTilt[anchorIndex] ?? DEFAULT_CAM_TILT;
  const delta = THREE.MathUtils.degToRad(tiltDeg - prev);
  camera.rotateX(-delta);
  appliedCamTilt[anchorIndex] = tiltDeg;
}

function attachCameraHelper(anchorIndex: number, camera: THREE.PerspectiveCamera) {
  if (!SHOW_ANCHOR_CAMERA_FRUSTUMS) return;
  const existing = cameraHelpers[anchorIndex];
  if (existing) {
    if (existing.camera === camera) return;
    scene.remove(existing);
    existing.dispose();
  }
  const helper = new THREE.CameraHelper(camera);
  scene.add(helper);
  cameraHelpers[anchorIndex] = helper;
}

function clearHover() {
  outlinePass.selectedObjects = [];
  currentHoverType = null;
  currentHoverIndex = -1;
  document.body.style.cursor = 'default';
}

function applyHover(targetMesh: THREE.Object3D, type: string, index: number) {
  if (currentHoverType === type && currentHoverIndex === index) return;
  clearHover();

  currentHoverType = type;
  currentHoverIndex = index;
  document.body.style.cursor = 'pointer';

    // OutlinePass natively handles traversing the hierarchy to draw the outline
    // without hacking any materials, preventing the shared material bug!
  outlinePass.selectedObjects = [targetMesh];
}

function isDescendant(child: THREE.Object3D, parent: THREE.Object3D): boolean {
  let current: THREE.Object3D | null = child;
  while (current) {
    if (current === parent) return true;
    current = current.parent;
  }
  return false;
}

window.addEventListener('pointermove', (event) => {
  if (event.target !== renderer.domElement) {
    clearHover();
    return;
  }

  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);

  const intersects = raycaster.intersectObjects(scene.children, true);

  let foundType: string | null = null;
  let foundIndex = -1;
  let hoverTarget: THREE.Object3D | null = null;

    // Check all intersections along the ray to bypass invisible room walls
  for (const hit of intersects) {
    const obj = hit.object;

    const gripperMesh = gripper.getInteractableMesh();
    if (gripperMesh && isDescendant(obj, gripperMesh)) {
      foundType = 'Gripper';
      foundIndex = 0;
      hoverTarget = gripperMesh;
      break;
    }

    for (let i = 0; i < corners.length; i++) {
      const corner = corners[i];
      if (corner instanceof ArpAnchor) {
        const anchorMesh = corner.getInteractableMesh();
        if (anchorMesh && isDescendant(obj, anchorMesh)) {
          foundType = 'Anchor';
          foundIndex = i;
          hoverTarget = anchorMesh;
          break;
        }
      }
    }

    // Check if hitting the room mesh (floor)
    if (obj === room.mesh) {
      // Verify if hit point is roughly at the floor plane (Y=0) because the mesh also includes the walls and celing.
      if (Math.abs(hit.point.y) < 0.05) {
        room.setReticule(hit.point);
      }
    }
  }

  if (foundType && hoverTarget) {
    applyHover(hoverTarget, foundType, foundIndex);
  } else {
    clearHover();
  }
});

window.addEventListener('click', (event) => {
  if (event.target !== renderer.domElement) return;
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(scene.children, true);
  if (currentHoverType) {
    openComponentPanel(currentHoverType, currentHoverIndex);
  } else {
    let hitFloor = false;
    for (const hit of intersects) {
      if (hit.object === room.mesh && Math.abs(hit.point.y) < 0.05) {
        targetListManager.showFloorGotoPopup(event.clientX, event.clientY, hit.point);
        hitFloor = true;
        break;
      }
    }
    if (!hitFloor) {
      targetListManager.hideGotoPopup();
    }
  }
});

function openComponentPanel(type: string, index: number) {
  const overlay = document.getElementById('component-details-overlay');
  const title = document.getElementById('cd-title');
  const ipEl = document.getElementById('cd-ip');
  const statusEl = document.getElementById('cd-status');
  const tempEl = document.getElementById('cd-temp');
  const motorTorqueEl = document.getElementById('cd-motor-torque');
  const sensorsEl = document.getElementById('cd-sensors');
  const anchorActions = document.getElementById('cd-anchor-actions');

  if (!overlay || !title || !ipEl || !statusEl || !tempEl || !motorTorqueEl || !sensorsEl || !anchorActions) return;

  let name = "";
  if (type === 'Anchor') {
    name = `Anchor ${index}`;
    anchorActions.style.display = 'flex';
    const camInput = document.getElementById('cd-cam-angle-input') as HTMLInputElement | null;
    if (camInput && lastTiltAngles[index] != null) {
      camInput.value = lastTiltAngles[index].toString();
    }
  } else {
    name = "Gripper";
    for (const key of componentStates.keys()) {
      if (componentStates.get(key)?.type === 'Gripper') {
        name = key;
        break;
      }
    }
    anchorActions.style.display = 'none';
  }

  activeComponentData = { type, index, name };
  title.textContent = `${name} Details`;

  const state = componentStates.get(name);
  ipEl.textContent = state?.ip || "Unknown";
  let connected = state?.status === nf.telemetry.ConnStatus.CONNSTATUS_CONNECTED;
  statusEl.textContent = connected ? "Connected" : "Disconnected";

  tempEl.textContent = state?.temp != null ? `${state.temp.toFixed(1)} °C` : "—";

  if (state?.motorTorque === nf.telemetry.MotorTorque.MOTORTORQUE_ENABLED) {
    motorTorqueEl.textContent = "Enabled";
  } else if (state?.motorTorque === nf.telemetry.MotorTorque.MOTORTORQUE_DISABLED) {
    motorTorqueEl.textContent = "Disabled";
  } else {
    motorTorqueEl.textContent = "—";
  }

  if (state?.errorMessage) {
    sensorsEl.textContent = state.errorMessage;
  } else if (connected) {
    sensorsEl.textContent = "sensors nominal";
  } else {
    sensorsEl.textContent = "";
  }

    // set 'disabled' state for all action buttons in this panel based on whether component is connected
  const actionButtons = overlay.querySelectorAll('.run-button');
  actionButtons.forEach(btn => {
    btn.classList.toggle('disabled', !connected);
  });

  overlay.classList.remove('hidden');
}

function initComponentDetailsPanel() {
  document.getElementById('cd-bg-catcher')?.addEventListener('click', () => {
    document.getElementById('component-details-overlay')?.classList.add('hidden');
    activeComponentData = null;
  });
  document.getElementById('close-component-panel')?.addEventListener('click', () => {
    document.getElementById('component-details-overlay')?.classList.add('hidden');
    activeComponentData = null;
  });
  document.getElementById('cd-ip-row')?.addEventListener('click', () => {
    const ip = document.getElementById('cd-ip')?.textContent ?? '';
    if (!ip || ip === 'Unknown') return;
    navigator.clipboard.writeText(ip).then(() => {
      const icon = document.querySelector<SVGElement>('#cd-ip-row .copy-icon');
      if (!icon) return;
      icon.classList.add('copied');
      setTimeout(() => icon.classList.remove('copied'), 1500);
    });
  });

    // Helper to bind buttons and respect the 'disabled' class
  const bindCdBtn = (id: string, actionFn: (type: string, index: number) => void) => {
    document.getElementById(id)?.addEventListener('click', (e) => {
      const btn = e.currentTarget as HTMLElement;
      if (!btn.classList.contains('disabled') && activeComponentData) {
        actionFn(activeComponentData.type, activeComponentData.index);
      }
    });
  };

  bindCdBtn('btn-cd-identify', handleComponentIdentify);

  bindCdBtn('btn-cd-tighten', (type, index) => {
    if (type === 'Anchor') handleAnchorTighten(index);
  });
  bindCdBtn('btn-cd-relax', (type, index) => {
    if (type === 'Anchor') handleAnchorRelax(index);
  });
  bindCdBtn('btn-cd-set-cam-angle', (type, index) => {
    if (type === 'Anchor') handleSetCamAngle(index);
  });
}


function handleComponentIdentify(type: string, index: number) {
  sendSingleComponentAction(type, index, nf.control.ComponentAction.COMPONENTACTION_IDENTIFY);
}

function handleAnchorTighten(index: number) {
  sendSingleComponentAction('Anchor', index, nf.control.ComponentAction.COMPONENTACTION_TIGHTEN);
}

function handleAnchorRelax(index: number) {
  sendSingleComponentAction('Anchor', index, nf.control.ComponentAction.COMPONENTACTION_RELAX);
}

function handleSetCamAngle(index: number) {
  const input = document.getElementById('cd-cam-angle-input') as HTMLInputElement;
  const angle = parseFloat(input?.value ?? String(DEFAULT_CAM_TILT));
  sendControl([nf.control.ControlItem.create({
    singleComponentAction: {
      isGripper: false,
      anchorNum: index,
      action: nf.control.ComponentAction.COMPONENTACTION_SET_CAM_ANGLE,
      camAngle: isNaN(angle) ? DEFAULT_CAM_TILT : angle,
    }
  })]);
}

initComponentDetailsPanel();