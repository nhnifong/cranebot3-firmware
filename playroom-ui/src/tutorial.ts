// ============================================================
// Guided tutorial overlay for the playroom.
//
// The tutorial is reached by visiting the "/tutorial" alias of the
// playroom (the backend serves the same playroom.html for that path).
// It always starts in LAN mode and walks the user through getting a
// robot online and driving it.
//
// The entire tutorial is described by the TUTORIAL_STEPS data structure
// below: each step names a message, an element to highlight, an optional
// copyable command block, and a predicate deciding when it auto-dismisses.
// Add or reorder steps there — the TutorialManager handles the rest.
// ============================================================

import { isFullyConnected, hasOperationCompleted, isSwingCancellationEnabled } from './main.ts';

export interface TutorialStep {
  /** Stable id, handy for debugging. */
  id: string;
  /** Explanation shown in the floating panel. May contain inline HTML. */
  message: string;
  /**
   * id of the element to highlight and anchor the panel next to. Omit for
   * steps that have no on-screen anchor (the panel floats near the top).
   */
  highlightId?: string;
  /**
   * Optional id of a menu/popup element (a `.run-menu-content`) to force open
   * while this step is active by adding the `show` class. Used when the
   * highlighted element lives inside a menu. Closed again when the step ends.
   */
  openMenuId?: string;
  /** Optional monospaced, copyable command block shown under the message. */
  copyBlock?: string;
  /**
   * Optional predicate. When it returns true the step auto-dismisses and the
   * tutorial advances. Evaluated by polling, so it should be cheap and based
   * on observable state (e.g. reading the DOM). Omit for manual-only steps.
   */
  dismissWhen?: () => boolean;
}

export const TUTORIAL_STEPS: TutorialStep[] = [

  { // Run stringman-headless on localhost and get online
    id: 'run-host',
    highlightId: 'system-status',
    message: 'Run <strong>stringman-headless</strong> in a virtualenv on this machine to begin:',
    copyBlock: [
      'python3 -m venv venv',
      'source venv/bin/activate',
      'pip install nf_robot[host]',
      'stringman-headless',
    ].join('\n'),
    // Auto-dismiss once the robot is online (status dot drops its offline class).
    dismissWhen: () => !document.getElementById('status-dot-el')?.classList.contains('status-offline'),
  },

  { // turn on the robot and detect the components
    id: 'connect-components',
    highlightId: 'component-status',
    message: 'Great job!<br><br>Now power on the robot. Show a <a href="/docs/arp_install_guide/#connect-all-components-to-wifi" target="_blank" rel="noopener">wifi share code</a> to each component\'s camera to connect it to your network. Check here for Anchors and Gripper to be detected by the motion controller.',
    dismissWhen: () => isFullyConnected(),
  },

  { // bring every component up to the latest firmware
    id: 'update-firmware',
    highlightId: 'action-update-firmware',
    openMenuId: 'maintenance-menu',
    message: 'Looks like everything is online<br><br>If this is a new install, update the components before using them: select <strong>"Update Firmware on Components"</strong> to run the upgrade on all connected anchors and the gripper.',
    dismissWhen: () => hasOperationCompleted('Update Component Firmware'),
  },

  { // attach your carabiners to the marker box
    id: 'carabiners',
    highlightId: 'action-disable-torque',
    openMenuId: 'maintenance-menu',
    message: `With the gripper in the center of the room, Attach all four lines with carabiners so no lines cross. In order to be able to pull out the lines while the robot is turned on, disable the torque.<br><br><img src="${import.meta.env.VITE_ASSET_BUCKET_URL ?? ''}/assets/playroom/simplified_box.webp">`,
  },

  { // Place markers on the floor
    id: 'markers',
    message: `Place the four large marker cards on the floor.<br<br>"origin" goes in the center<br><br><img src="${import.meta.env.VITE_ASSET_BUCKET_URL ?? ''}/assets/playroom/cards_on_floor.webp">`,
  },

  { // run calibration
    id: 'run-calibration',
    highlightId: 'action-full-cal',
    openMenuId: 'maintenance-menu',
    message: `Select <strong>"Full Calibration"</strong>`,
    dismissWhen: () => hasOperationCompleted('Calibration'),
  },

  { // enable swing cancellation
    id: 'swing-cancellation',
    highlightId: 'btn-swing-cancel',
    message: `Enable swing cancellation for smoother movement`,
    dismissWhen: () => isSwingCancellationEnabled(),
  },

];

export function isTutorialMode(): boolean {
  return /(^|\/)tutorial\/?$/.test(window.location.pathname);
}

class TutorialManager {
  private steps: TutorialStep[];
  private index = 0;
  private panel: HTMLElement | null = null;
  private highlighted: HTMLElement | null = null;
  private openedMenu: HTMLElement | null = null;
  private endBtn: HTMLElement | null = null;
  private pollTimer: number | null = null;
  // True when the user manually hid the current step's message. We stay on the
  // step and keep polling its predicate; only the predicate advances steps.
  private dismissed = false;
  private repositionHandler = () => this.positionPanel();

  constructor(steps: TutorialStep[]) {
    this.steps = steps;
  }

  start() {
    this.injectEndButton();
    this.index = 0;
    this.showCurrentStep();

    // Poll the current step's predicate. Advancing to the next step happens
    // only when the predicate becomes true — never from a manual dismiss. While
    // the message is still showing keep it anchored to its target.
    this.pollTimer = window.setInterval(() => {
      const step = this.steps[this.index];
      if (!step) return; // ran out of steps; tutorial stays active until End.
      if (step.dismissWhen?.()) {
        this.advance();
      } else if (!this.dismissed) {
        // Re-assert the menu in case a stray click elsewhere closed it.
        this.openedMenu?.classList.add('show');
        this.positionPanel();
      }
    }, 400);

    window.addEventListener('resize', this.repositionHandler);
    window.addEventListener('scroll', this.repositionHandler, true);
  }

  /** Tear down everything and restore the URL back to /playroom. */
  end = () => {
    this.clearPanel();
    if (this.pollTimer !== null) {
      window.clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    window.removeEventListener('resize', this.repositionHandler);
    window.removeEventListener('scroll', this.repositionHandler, true);
    this.endBtn?.remove();
    this.endBtn = null;

    // Swap /tutorial back to /playroom, keeping any query params (robotid=lan).
    const url = new URL(window.location.href);
    url.pathname = url.pathname.replace(/(^|\/)tutorial\/?$/, '$1playroom');
    window.history.replaceState({}, '', url);
  };

  // --- Step lifecycle ---------------------------------------------------

  private showCurrentStep() {
    this.clearPanel();
    this.dismissed = false;
    const step = this.steps[this.index];
    if (!step) return; // ran out of steps; tutorial stays active until End.

    // Force open any menu the highlighted element lives inside.
    if (step.openMenuId) {
      const menu = document.getElementById(step.openMenuId);
      menu?.classList.add('show');
      this.openedMenu = menu;
    }

    // Steps without a highlightId have no anchor; the panel floats near the top.
    if (step.highlightId) {
      const target = document.getElementById(step.highlightId);
      if (target) {
        target.classList.add('tutorial-highlight');
        this.highlighted = target;
      } else {
        console.warn(`Tutorial step "${step.id}": no element with id "${step.highlightId}"`);
      }
    }

    this.panel = this.buildPanel(step);
    document.body.appendChild(this.panel);
    this.positionPanel();
  }

  /** Move to the next step. Driven only by the current step's predicate. */
  private advance = () => {
    this.index++;
    this.showCurrentStep();
  };

  /**
   * Hide the current step's message and unhighlight its target, but stay on the
   * step — the predicate still decides when we advance. A step with no
   * predicate has nothing to wait on, so a manual dismiss advances it instead.
   */
  private dismissPanel = () => {
    if (this.steps[this.index]?.dismissWhen) {
      this.clearPanel();
      this.dismissed = true;
    } else {
      this.advance();
    }
  };

  private clearPanel() {
    if (this.highlighted) {
      this.highlighted.classList.remove('tutorial-highlight');
      this.highlighted = null;
    }
    if (this.openedMenu) {
      this.openedMenu.classList.remove('show');
      this.openedMenu = null;
    }
    this.panel?.remove();
    this.panel = null;
  }

  // --- DOM construction -------------------------------------------------

  private buildPanel(step: TutorialStep): HTMLElement {
    const panel = document.createElement('div');
    panel.className = 'tutorial-panel';

    const dismiss = document.createElement('button');
    dismiss.className = 'tutorial-panel-dismiss';
    dismiss.title = 'Dismiss this message';
    dismiss.textContent = '✕';
    dismiss.addEventListener('click', this.dismissPanel);
    panel.appendChild(dismiss);

    const msg = document.createElement('div');
    msg.className = 'tutorial-panel-message';
    msg.innerHTML = step.message;
    panel.appendChild(msg);

    if (step.copyBlock) {
      panel.appendChild(this.buildCopyBlock(step.copyBlock));
    }

    return panel;
  }

  private buildCopyBlock(text: string): HTMLElement {
    const wrap = document.createElement('div');
    wrap.className = 'tutorial-copy-block';

    const pre = document.createElement('pre');
    pre.textContent = text;
    wrap.appendChild(pre);

    const copyBtn = document.createElement('button');
    copyBtn.className = 'tutorial-copy-btn';
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = 'Copied!';
        window.setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
      } catch {
        copyBtn.textContent = 'Copy failed';
      }
    });
    wrap.appendChild(copyBtn);

    return wrap;
  }

  private injectEndButton() {
    const header = document.querySelector('#panel-header .header-left');
    if (!header) return;
    const btn = document.createElement('button');
    btn.id = 'btn-end-tutorial';
    btn.className = 'header-back-btn tutorial-end-btn';
    btn.title = 'Exit the tutorial';
    btn.textContent = 'End Tutorial';
    btn.addEventListener('click', this.end);
    header.appendChild(btn);
    this.endBtn = btn;
  }

  // --- Positioning ------------------------------------------------------

  /** Anchor the panel next to its highlighted target, preferring above. */
  private positionPanel() {
    if (!this.panel) return;
    const panel = this.panel.getBoundingClientRect();
    const margin = 8;

    // No anchor element (missing/typo'd highlightId): float near the top
    // center so the message is still visible rather than silently lost.
    if (!this.highlighted) {
      this.panel.style.top = `${margin * 2}px`;
      this.panel.style.left = `${Math.max(margin, (window.innerWidth - panel.width) / 2)}px`;
      return;
    }

    const target = this.highlighted.getBoundingClientRect();
    const gap = 14;

    // Vertical: prefer above the target, fall back to below.
    let top = target.top - panel.height - gap;
    if (top < margin) top = target.bottom + gap;
    top = Math.min(top, window.innerHeight - panel.height - margin);
    top = Math.max(margin, top);

    // Horizontal: align to the target's left, clamped to the viewport.
    let left = target.left;
    left = Math.min(left, window.innerWidth - panel.width - margin);
    left = Math.max(margin, left);

    this.panel.style.top = `${top}px`;
    this.panel.style.left = `${left}px`;
  }
}

let manager: TutorialManager | null = null;

/** Start the tutorial if the page was reached via the /tutorial alias. */
export function maybeStartTutorial() {
  if (!isTutorialMode() || manager) return;
  manager = new TutorialManager(TUTORIAL_STEPS);
  manager.start();
}
