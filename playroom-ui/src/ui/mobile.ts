/**
 * Mobile shell for the Stringman control panel.
 *
 * Responsibilities:
 *   - Tab navigation (3D / Cams / Targets / Drive / System)
 *   - Persistent topbar that mirrors status + op-progress and hosts STOP
 *   - Locks the 3D camera to its initial pose on mobile (no orbit/pan/zoom)
 *   - Simulated joystick + finger/wrist strips that feed the existing
 *     GamepadController input-merge pipeline
 *   - Moves the selected video feed into the Drive tab when the user taps
 *     a feed's "Drive" button, and returns it to #panel-right on tab switch
 *
 * --------------------------------------------------------------------
 * How "mobile mode" is detected:
 *   `?mobile=1` URL param          → force on (great for desktop testing)
 *   `?mobile=0` URL param          → force off
 *   otherwise: (max-width: 820px) AND (pointer: coarse) → on
 * The decision is made once at startup by `MobileShell.shouldEnable()`.
 *
 * Mobile mode toggles a single class on <body> and <html>:
 *   - body.mobile          — gates ALL mobile CSS in mobile.css
 *   - html.mobile-locked   — locks the document to the visible viewport
 *
 * --------------------------------------------------------------------
 * To make something happen on mobile only:
 *
 *   In CSS:    write rules prefixed with `body.mobile ...`.
 *              Use `.mobile-only` on elements that should only render
 *              on mobile (it's already wired to display:flex under
 *              body.mobile and display:none otherwise).
 *
 *   In TS:     gate with `document.body.classList.contains('mobile')`.
 *              Don't call `MobileShell.shouldEnable()` at runtime — it
 *              re-runs media queries and may give a stale answer if the
 *              user resized the window. The body class is the single
 *              source of truth after startup.
 *
 *   For input: don't add a new merge path in gamepad.ts. Instead, mutate
 *              this.state and let `GamepadController.touchProvider` pull
 *              it. The merge already understands all 16 button channels
 *              + two analog sticks + analog triggers.
 *
 * --------------------------------------------------------------------
 * Design notes worth keeping in mind:
 *
 *   1. The Three.js canvas is appended to <body> by main.ts with no
 *      positioning, so on phones (where the URL bar shrinks dynamically)
 *      its in-flow height was pushing the body taller than the visible
 *      area and hiding the fixed tab bar. mobile.css fixes this by
 *      pinning canvas + body to `100vw × 100dvh` with overflow:hidden.
 *      If you ever change how the canvas is mounted, revisit that.
 *
 *   2. The Drive tab borrows the selected video element (it moves the
 *      DOM node rather than instantiating a second feed) so WebRTC
 *      connections, target-overlay canvases, and listeners ride along.
 *      `returnDriveFeed()` puts it back into #panel-right in its original
 *      DOM order whenever the user leaves the Drive tab.
 *
 *   3. The simulated joystick / finger / wrist strips translate analog
 *      drags back into the button-hold model gamepad.ts expects (e.g.
 *      finger strip above center → buttons.b held). This keeps the
 *      existing slow→fast finger-speed ramp and rising-edge dispatch
 *      logic working without any changes to that file.
 *
 *   4. While a text input is focused, getTouchInputState() returns null
 *      — mirrors the keyboard guard in gamepad.ts so typing into the
 *      LeRobot dataset name / debug command / share email doesn't move
 *      the robot via accidental finger contact.
 *
 *   5. OrbitControls is disabled on mobile (locked perspective per
 *      product decision). Tap-to-go on the 3D tab still works because
 *      it goes through main.ts's `click` listener on renderer.domElement,
 *      which doesn't need OrbitControls.
 */

import type * as THREE from 'three';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/** Same shape as GamepadController.getKeyboardState() / getGamepadState() output. */
export interface TouchInputState {
  leftStick: { x: number; y: number };
  rightStick: { x: number; y: number };
  buttons: {
    a: boolean; b: boolean; x: boolean; y: boolean;
    lb: boolean; rb: boolean;
    lt: number; rt: number;
    select: boolean; start: boolean;
    lclick: boolean; rclick: boolean;
    dpadUp: boolean; dpadDown: boolean; dpadLeft: boolean; dpadRight: boolean;
  };
}

export type Tab = '3d' | 'cams' | 'targets' | 'drive' | 'system';

export interface MobileOptions {
  controls: OrbitControls;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  /** Set the perspective (matches main.ts perspective enum values). */
  setPerspective: (mode: number) => void;
  perspectives: { top: number; bottom: number; gripper: number };
}

const TAB_CLASSES: Record<Tab, string> = {
  '3d': 'tab-3d',
  cams: 'tab-cams',
  targets: 'tab-targets',
  drive: 'tab-drive',
  system: 'tab-system',
};

function neutralTouchState(): TouchInputState {
  return {
    leftStick: { x: 0, y: 0 },
    rightStick: { x: 0, y: 0 },
    buttons: {
      a: false, b: false, x: false, y: false,
      lb: false, rb: false,
      lt: 0, rt: 0,
      select: false, start: false,
      lclick: false, rclick: false,
      dpadUp: false, dpadDown: false, dpadLeft: false, dpadRight: false,
    },
  };
}

export class MobileShell {
  private state: TouchInputState = neutralTouchState();
  private opts: MobileOptions;
  private currentDriveFeedId: string | null = null;

  constructor(opts: MobileOptions) {
    this.opts = opts;

    if (!MobileShell.shouldEnable()) return;

    document.body.classList.add('mobile');
    document.documentElement.classList.add('mobile-locked');
    this.measureBars();
    this.setTab('3d');

    // Re-fit the renderer/canvas after mobile CSS is applied, in case the
    // initial size was computed against a body that included flow content.
    requestAnimationFrame(() => {
      window.dispatchEvent(new Event('resize'));
    });

    this.installTabBar();
    this.installTopBar();
    this.installDriveButtons();
    this.installJoystick();
    this.installVerticalButtons();
    this.installWristStrip();
    this.installFingerStrip();
    this.installSecondaryButtons();
    this.installMirrors();
    this.lockCamera();

    window.addEventListener('resize', () => this.measureBars());
    window.addEventListener('orientationchange', () => {
      // Re-measure after the orientation transition completes.
      setTimeout(() => this.measureBars(), 250);
    });

    console.log('[mobile] shell enabled');
  }

  /** Whether the page should be in mobile mode. */
  public static shouldEnable(): boolean {
    const params = new URLSearchParams(window.location.search);
    if (params.get('mobile') === '1') return true;
    if (params.get('mobile') === '0') return false;
    const narrow = window.matchMedia('(max-width: 820px)').matches;
    const coarse = window.matchMedia('(pointer: coarse)').matches;
    return narrow && coarse;
  }

  /** Pulled by GamepadController each frame. Returns null if not on mobile. */
  public getTouchInputState(): TouchInputState | null {
    if (!document.body.classList.contains('mobile')) return null;
    // Mirror the keyboard guard in gamepad.ts: don't drive while a text
    // field is focused (e.g. typing a LeRobot dataset name).
    const active = document.activeElement;
    if (active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement) {
      return null;
    }
    return this.state;
  }

  // ================== Tab navigation ==================

  public setTab(tab: Tab) {
    for (const t of Object.keys(TAB_CLASSES) as Tab[]) {
      document.body.classList.toggle(TAB_CLASSES[t], t === tab);
    }
    document.querySelectorAll('.tabbar-btn').forEach((btn) => {
      const el = btn as HTMLElement;
      el.classList.toggle('active', el.dataset.tab === tab);
    });

    // Camera/Orbit gestures only ever active on the 3D tab — and even then,
    // disabled on mobile per product direction (locked perspective).
    this.lockCamera();

    // When leaving Drive tab, return the borrowed feed to its home panel.
    if (tab !== 'drive') this.returnDriveFeed();
  }

  private installTabBar() {
    document.querySelectorAll('.tabbar-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const t = (btn as HTMLElement).dataset.tab as Tab | undefined;
        if (!t) return;
        if ((btn as HTMLButtonElement).disabled) return;
        this.setTab(t);
      });
    });
  }

  private measureBars() {
    const top = document.getElementById('mobile-topbar');
    const bot = document.getElementById('mobile-tabbar');
    if (top) document.documentElement.style.setProperty('--mobile-topbar-h', `${top.offsetHeight}px`);
    if (bot) document.documentElement.style.setProperty('--mobile-tabbar-h', `${bot.offsetHeight}px`);
  }

  // ================== Top bar mirroring ==================

  private installTopBar() {
    // Mobile STOP button delegates to the desktop one (which carries all the
    // wiring already). Falls back to clicking by ID if not yet bound.
    document.getElementById('mobile-stop-btn')?.addEventListener('click', (e) => {
      e.preventDefault();
      document.getElementById('stop-btn')?.click();
    });
    // Mobile Change Robot button delegates to the desktop header button.
    document.getElementById('mobile-change-robot-btn')?.addEventListener('click', (e) => {
      e.preventDefault();
      document.getElementById('btn-header-back')?.click();
    });
  }

  private installMirrors() {
    // Mirror status text + dot
    const statusText = document.getElementById('status-text');
    const statusDot = document.getElementById('status-dot-el');
    const mStatusText = document.getElementById('mobile-status-text');
    const mStatusDot = document.getElementById('mobile-status-dot');

    const syncStatus = () => {
      if (statusText && mStatusText) mStatusText.textContent = statusText.textContent;
      if (statusDot && mStatusDot) {
        mStatusDot.classList.toggle('status-offline', statusDot.classList.contains('status-offline'));
      }
    };
    syncStatus();
    if (statusText) new MutationObserver(syncStatus).observe(statusText, { childList: true, characterData: true, subtree: true });
    if (statusDot) new MutationObserver(syncStatus).observe(statusDot, { attributes: true, attributeFilter: ['class'] });

    // Mirror op progress
    const opContainer = document.getElementById('op-progress-container');
    const opName = document.getElementById('op-name');
    const opAction = document.getElementById('op-action');
    const opFill = document.getElementById('op-bar-fill');
    const mOpContainer = document.getElementById('mobile-op-progress');
    const mOpName = document.getElementById('mobile-op-name');
    const mOpAction = document.getElementById('mobile-op-action');
    const mOpFill = document.getElementById('mobile-op-bar-fill');

    const syncOp = () => {
      if (!opContainer || !mOpContainer) return;
      const visible = !opContainer.classList.contains('hidden');
      mOpContainer.classList.toggle('hidden', !visible);
      if (mOpName && opName) mOpName.textContent = opName.textContent;
      if (mOpAction && opAction) mOpAction.textContent = opAction.textContent;
      if (mOpFill && opFill) mOpFill.style.width = (opFill as HTMLElement).style.width;
    };
    syncOp();
    if (opContainer) new MutationObserver(syncOp).observe(opContainer, { attributes: true, childList: true, subtree: true, characterData: true });
  }

  // ================== Drive entry from a video feed ==================

  private installDriveButtons() {
    const map: Array<{ feedId: string; perspective: number }> = [
      { feedId: 'firstOverhead', perspective: this.opts.perspectives.top },
      { feedId: 'secondOverhead', perspective: this.opts.perspectives.bottom },
      { feedId: 'gripper', perspective: this.opts.perspectives.gripper },
    ];
    for (const { feedId, perspective } of map) {
      const feed = document.getElementById(feedId);
      const btn = feed?.querySelector('.feed-drive-btn');
      if (!btn) continue;
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.enterDrive(feedId, perspective);
      });
    }
  }

  private enterDrive(feedId: string, perspective: number) {
    const feed = document.getElementById(feedId);
    const host = document.getElementById('drive-feed-host');
    if (!feed || !host) return;

    // Move the feed element into the drive host. Listeners and video state
    // ride along since we're moving the same DOM node.
    if (this.currentDriveFeedId && this.currentDriveFeedId !== feedId) this.returnDriveFeed();
    if (feed.parentElement !== host) host.appendChild(feed);
    this.currentDriveFeedId = feedId;

    // Enable Drive tab and switch.
    const driveTab = document.querySelector('.tabbar-btn-drive') as HTMLButtonElement | null;
    if (driveTab) driveTab.disabled = false;
    this.opts.setPerspective(perspective);
    this.setTab('drive');
  }

  /** Return the borrowed feed to its home panel (#panel-right). */
  private returnDriveFeed() {
    if (!this.currentDriveFeedId) return;
    const feed = document.getElementById(this.currentDriveFeedId);
    const home = document.getElementById('panel-right');
    if (feed && home && feed.parentElement !== home) {
      // Preserve original DOM order by id. Insert before next sibling that
      // appears later in ALL_FEED_IDS, otherwise append.
      const order = ['firstOverhead', 'secondOverhead', 'gripper'];
      const myIdx = order.indexOf(this.currentDriveFeedId);
      let inserted = false;
      for (let i = myIdx + 1; i < order.length; i++) {
        const sibling = home.querySelector(`#${order[i]}`);
        if (sibling) {
          home.insertBefore(feed, sibling);
          inserted = true;
          break;
        }
      }
      if (!inserted) home.appendChild(feed);
    }
    this.currentDriveFeedId = null;
    const driveTab = document.querySelector('.tabbar-btn-drive') as HTMLButtonElement | null;
    if (driveTab) driveTab.disabled = true;
  }

  // ================== Camera lock ==================

  private lockCamera() {
    // Mobile direction: lock 3D view to its initial perspective. Disable all
    // OrbitControls gestures unconditionally on mobile.
    if (document.body.classList.contains('mobile')) {
      this.opts.controls.enabled = false;
    } else {
      this.opts.controls.enabled = true;
    }
  }

  // ================== Joystick (left stick) ==================

  private installJoystick() {
    const stick = document.getElementById('drive-stick');
    const knob = stick?.querySelector('.touch-stick-knob') as HTMLElement | undefined;
    if (!stick || !knob) return;

    let activePointerId: number | null = null;
    let centerX = 0, centerY = 0, radius = 0;

    const recenter = () => {
      const rect = stick.getBoundingClientRect();
      centerX = rect.left + rect.width / 2;
      centerY = rect.top + rect.height / 2;
      radius = rect.width / 2 - 32; // knob radius padding
    };

    const updateKnob = (dx: number, dy: number) => {
      const mag = Math.sqrt(dx * dx + dy * dy);
      let nx = dx, ny = dy;
      if (mag > radius) {
        nx = (dx / mag) * radius;
        ny = (dy / mag) * radius;
      }
      knob.style.transform = `translate(calc(-50% + ${nx}px), calc(-50% + ${ny}px))`;
      // Map to [-1, 1]; invert Y (up = positive)
      this.state.leftStick.x = nx / radius;
      this.state.leftStick.y = -ny / radius;
    };

    const reset = () => {
      knob.style.transform = `translate(-50%, -50%)`;
      this.state.leftStick.x = 0;
      this.state.leftStick.y = 0;
      stick.classList.remove('active');
      activePointerId = null;
    };

    stick.addEventListener('pointerdown', (e) => {
      if (activePointerId !== null) return;
      activePointerId = e.pointerId;
      stick.setPointerCapture(e.pointerId);
      stick.classList.add('active');
      recenter();
      updateKnob(e.clientX - centerX, e.clientY - centerY);
      e.preventDefault();
    });
    stick.addEventListener('pointermove', (e) => {
      if (e.pointerId !== activePointerId) return;
      updateKnob(e.clientX - centerX, e.clientY - centerY);
    });
    const release = (e: PointerEvent) => {
      if (e.pointerId !== activePointerId) return;
      reset();
    };
    stick.addEventListener('pointerup', release);
    stick.addEventListener('pointercancel', release);
    stick.addEventListener('lostpointercapture', () => reset());
  }

  // ================== Wrist strip (right stick X) ==================

  private installWristStrip() {
    const strip = document.getElementById('drive-wrist');
    const knob = strip?.querySelector('.touch-strip-knob') as HTMLElement | undefined;
    if (!strip || !knob) return;

    let activePointerId: number | null = null;
    let centerX = 0, halfWidth = 0;

    const recenter = () => {
      const rect = strip.getBoundingClientRect();
      centerX = rect.left + rect.width / 2;
      halfWidth = rect.width / 2 - 30;
    };

    const updateKnob = (clientX: number) => {
      let dx = clientX - centerX;
      if (dx > halfWidth) dx = halfWidth;
      if (dx < -halfWidth) dx = -halfWidth;
      knob.style.transform = `translate(calc(-50% + ${dx}px), 0)`;
      this.state.rightStick.x = dx / halfWidth;
    };
    const reset = () => {
      knob.style.transform = `translateX(-50%)`;
      this.state.rightStick.x = 0;
      strip.classList.remove('active');
      activePointerId = null;
    };

    strip.addEventListener('pointerdown', (e) => {
      if (activePointerId !== null) return;
      activePointerId = e.pointerId;
      strip.setPointerCapture(e.pointerId);
      strip.classList.add('active');
      recenter();
      updateKnob(e.clientX);
      e.preventDefault();
    });
    strip.addEventListener('pointermove', (e) => {
      if (e.pointerId !== activePointerId) return;
      updateKnob(e.clientX);
    });
    const release = (e: PointerEvent) => {
      if (e.pointerId !== activePointerId) return;
      reset();
    };
    strip.addEventListener('pointerup', release);
    strip.addEventListener('pointercancel', release);
    strip.addEventListener('lostpointercapture', () => reset());
  }

  // ================== Press-and-hold buttons ==================

  /** Wire a button so that pressing it sets `field` true on `target` until release. */
  private bindHold(elId: string, onPress: () => void, onRelease: () => void) {
    const el = document.getElementById(elId);
    if (!el) return;
    let active = false;
    const press = (e: Event) => {
      e.preventDefault();
      if (active) return;
      active = true;
      el.classList.add('pressed');
      onPress();
    };
    const release = () => {
      if (!active) return;
      active = false;
      el.classList.remove('pressed');
      onRelease();
    };
    el.addEventListener('pointerdown', (e) => {
      (el as HTMLElement).setPointerCapture((e as PointerEvent).pointerId);
      press(e);
    });
    el.addEventListener('pointerup', release);
    el.addEventListener('pointercancel', release);
    el.addEventListener('lostpointercapture', release);
  }

  private installVerticalButtons() {
    // Up → rt analog 1.0 (the gamepad code uses rt-lt for vertical)
    this.bindHold('drive-btn-up',
      () => { this.state.buttons.rt = 1; },
      () => { this.state.buttons.rt = 0; });
    // Down → lt analog 1.0
    this.bindHold('drive-btn-down',
      () => { this.state.buttons.lt = 1; },
      () => { this.state.buttons.lt = 0; });
  }

  /**
   * Vertical finger strip: drag the knob up to OPEN (sets buttons.b)
   * or down to CLOSE (sets buttons.a). Returns to center on release.
   *
   * We translate the analog drag back into the button-hold model that
   * `gamepad.ts` expects so the existing slow→fast ramp keeps working.
   * A small deadzone prevents jitter.
   */
  private installFingerStrip() {
    const strip = document.getElementById('drive-fingers');
    const knob = strip?.querySelector('.touch-strip-knob') as HTMLElement | undefined;
    if (!strip || !knob) return;

    let activePointerId: number | null = null;
    let centerY = 0, halfHeight = 0;
    const DEADZONE = 0.15; // fraction of half-height

    const recenter = () => {
      const rect = strip.getBoundingClientRect();
      centerY = rect.top + rect.height / 2;
      halfHeight = rect.height / 2 - 30;
    };

    const updateKnob = (clientY: number) => {
      let dy = clientY - centerY;
      if (dy > halfHeight) dy = halfHeight;
      if (dy < -halfHeight) dy = -halfHeight;
      knob.style.transform = `translateY(calc(-50% + ${dy}px))`;
      const frac = halfHeight > 0 ? dy / halfHeight : 0;
      // Above center (negative dy) = open; below center = close.
      // Apply a small deadzone so resting near center doesn't drift.
      const isOpen = frac < -DEADZONE;
      const isClose = frac > DEADZONE;
      this.state.buttons.b = isOpen;   // open → b (gamepad mapping)
      this.state.buttons.a = isClose;  // close → a
    };

    const reset = () => {
      knob.style.transform = `translateY(-50%)`;
      this.state.buttons.a = false;
      this.state.buttons.b = false;
      strip.classList.remove('active');
      activePointerId = null;
    };

    strip.addEventListener('pointerdown', (e) => {
      if (activePointerId !== null) return;
      activePointerId = e.pointerId;
      strip.setPointerCapture(e.pointerId);
      strip.classList.add('active');
      recenter();
      updateKnob(e.clientY);
      e.preventDefault();
    });
    strip.addEventListener('pointermove', (e) => {
      if (e.pointerId !== activePointerId) return;
      updateKnob(e.clientY);
    });
    const release = (e: PointerEvent) => {
      if (e.pointerId !== activePointerId) return;
      reset();
    };
    strip.addEventListener('pointerup', release);
    strip.addEventListener('pointercancel', release);
    strip.addEventListener('lostpointercapture', () => reset());
  }

  private installSecondaryButtons() {
    // Stop all → fires a single rising-edge `select` so the existing
    // gamepad code emits COMMAND_STOP_ALL.
    document.getElementById('drive-btn-stop-all')?.addEventListener('click', () => {
      this.state.buttons.select = true;
      // Release on the next frame so it's a single rising edge.
      setTimeout(() => { this.state.buttons.select = false; }, 50);
    });
  }
}
