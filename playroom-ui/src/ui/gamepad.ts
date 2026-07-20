import * as THREE from 'three';
import { nf } from '../generated/proto_bundle.js';
import { TargetListManager } from './target_list_manager.js';
import { Say } from '../utils.ts';

export class GamepadController {
    private gamepadIndex: number | null = null;

    // Constants
    private readonly DEADZONE = 0.1; // 10% deadzone
    private readonly GAMEPAD_GRIP_SLOW = 40; // deg per second
    private readonly GAMEPAD_GRIP_FAST = 90; // deg per second
    private readonly GAMEPAD_WRIST_DEG_PER_SEC = 150;


    // State Tracking
    public seatOrbitMode = true;
    private orbitObj: THREE.Object3D | null = null;
    private robotPosition: { x: number, y: number } | null = null;

    private lastSendT = 0;

    private seenValidTriggers = false;

    // Sometimes all you have are discrete trigger buttons that can output 0 or 1.
    // in that case let the vertical input be an EMA of RT-LT
    private useRampForTriggers = false;
    private rampedTriggerEMA = 0;
    
    // Previous Button States (Rising Edge Detection)
    private startWasHeld = false;
    private dpadUpWasHeld = false;
    private dpadLeftWasHeld = false;
    private dpadRightWasHeld = false;
    private selectWasHeld = false;
    private lclickWasHeld = false;
    private aWasHeld = false
    private bWasHeld = false;
    private yWasHeld = false;

    // time of a and b presses for speed ramp up
    private aPressTimestamp = 0;
    private bPressTimestamp = 0;

    // keyboard input state
    private keys = new Set<string>();
    private keyStates: { [code: string]: boolean } = {};
    private arrowPressTimestamps: { [code: string]: number } = {};

    // Change Detection (Store last "Action" vector: [vx, vy, vz, speed, finger, wrist])
    private lastAction = new Float32Array(6);

    public targetListManager: TargetListManager | null = null;

    public toggleSwingC: () => void = () => {};
    public onSetPrompt: () => void = () => {};

    // Optional provider for touch-derived input state (mobile shell).
    // Returns null when not on mobile or no input is being given.
    public touchProvider: (() => any | null) | null = null;

    constructor() {
        // Listen for connection events to know which index to poll
        window.addEventListener("gamepadconnected", (e) => {
            console.log("Gamepad connected at index %d: %s. %d buttons, %d axes.",
                e.gamepad.index, e.gamepad.id,
                e.gamepad.buttons.length, e.gamepad.axes.length);
            
            // Just grab the first one that connects
            if (this.gamepadIndex === null) {
                this.gamepadIndex = e.gamepad.index;
                this.seenValidTriggers = false;

                // Show the explanatory message
                const container = document.getElementById('how-to');
                if (container) {
                    container.textContent = "To activate gamepad, press both analog triggers.";
                }
            }
        });

        window.addEventListener("gamepaddisconnected", (e) => {
            console.log("Gamepad disconnected from index %d", e.gamepad.index);
            if (this.gamepadIndex === e.gamepad.index) {
                this.gamepadIndex = null;
            }
        });

        window.addEventListener('keydown', (e) => this.handleKey(e, true));
        window.addEventListener('keyup', (e) => this.handleKey(e, false));


        const container = document.getElementById('how-to');
        if (container && !document.body.classList.contains('mobile')) {
            container.textContent = "Use WASDQE to move or connect a gamepad. Space-LShift to grasp. Arrows for wrist/fingers. Z to set prompt.";
        }
    }

    private handleKey(e: KeyboardEvent, isDown: boolean) {
        // Don't control robot if user is typing in a text field
        const active = document.activeElement;
        if (active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement) {
            return;
        }

        const container = document.getElementById('how-to');
        if (container) {
            container.textContent = "";
        }

        if (isDown) {
            if (!this.keys.has(e.code) && ['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.code)) {
                this.arrowPressTimestamps[e.code] = Date.now() / 1000;
            }
            this.keys.add(e.code);
            if (e.code === "Delete" && this.targetListManager) {
                this.targetListManager.deleteSelectedItem();
            }
        } else {
            this.keys.delete(e.code);
            this.keyStates[e.code] = false; // Reset rising edge state on release
        }
    }

    /**
     * Returns null if no gamepad is active.
     * https://hardwaretester.com/gamepad is really helpful in debugging this
     */
    public getGamepadState() {
        if (this.gamepadIndex === null) return null;

        const gamepads = navigator.getGamepads();
        const gp = gamepads[this.gamepadIndex];

        if (!gp) return null;

        // Find the analog triggers
        let lt = 0.0;
        let rt = 0.0;
        if (gp.axes.length == 6) {
            // firefox
            lt = gp.axes[4]; 
            rt = gp.axes[4];
        } else {
            // chrome
            lt = gp.buttons[6].value;
            rt = gp.buttons[7].value;
        }

        return {
            leftStick: {
                x: this.applyDeadzone(gp.axes[0]), // positive is to the right
                y: this.applyDeadzone(-gp.axes[1]), // positive is up (after the negation)
            },
            rightStick: {
                x: this.applyDeadzone(gp.axes[2]),
                y: this.applyDeadzone(-gp.axes[3])
            },
            buttons: {
                // Standard mappings (0-3)
                a: gp.buttons[0].pressed,
                b: gp.buttons[1].pressed,
                y: gp.buttons[2].pressed, // not always labelled y, but usually at the top
                x: gp.buttons[3].pressed,
                // Bumpers (4-5)
                lb: gp.buttons[4].pressed,
                rb: gp.buttons[5].pressed,
                // Triggers - Value is 0.0 to 1.0
                lt: lt,
                rt: rt,
                // Extras
                select: gp.buttons[8]?.pressed || false, // somtimes also called back
                start: gp.buttons[9]?.pressed || false,
                // Stick clickers
                lclick: gp.buttons[10].pressed,
                rclick: gp.buttons[11].pressed,
                // D-Pad (Standard Indices 12-15)
                dpadUp: gp.buttons[12]?.pressed || false,
                dpadDown: gp.buttons[13]?.pressed || false,
                dpadLeft: gp.buttons[14]?.pressed || false,
                dpadRight: gp.buttons[15]?.pressed || false
            }
        };
    }

    /**
     * Returns a format identical to getGamepadState but using keyboard input.
     */
    public getKeyboardState() {
        return {
            leftStick: {
                x: (this.keys.has('KeyA') ? -1 : 0) + (this.keys.has('KeyD') ? 1 : 0),
                y: (this.keys.has('KeyS') ? -1 : 0) + (this.keys.has('KeyW') ? 1 : 0)
            },
            rightStick: {
                x: -this.arrowRamp('ArrowLeft') + this.arrowRamp('ArrowRight'),
                y: -this.arrowRamp('ArrowDown') + this.arrowRamp('ArrowUp')
            },
            buttons: {
                // Standard mappings (0-3)
                a: this.keys.has('Space'),
                b: this.keys.has('ShiftLeft'),
                y: this.keys.has('KeyZ'),
                x: this.keys.has('KeyX'),
                // Bumpers (4-5)
                lb: false,
                rb: false,
                // Triggers - Value is 0.0 to 1.0
                lt: this.keys.has('KeyQ') ? 1 : 0,
                rt: this.keys.has('KeyE') ? 1 : 0,
                // Extras
                select: this.keys.has('Backspace'),
                start: this.keys.has('Enter'),
                // Stick clickers
                lclick: this.keys.has('Digit5'),
                rclick: this.keys.has('Digit6'),
                // D-Pad (Standard Indices 12-15)
                dpadUp: this.keys.has('Digit1'),
                dpadDown:  this.keys.has('Digit4'),
                dpadLeft: this.keys.has('Digit2'),
                dpadRight: this.keys.has('Digit3')
            }
        };
    }

    private pressRamp(heldSince: number, slowVal: number, fastVal: number, rampAfter: number): number {
        return (Date.now() / 1000 - heldSince) >= rampAfter ? fastVal : slowVal;
    }

    private arrowRamp(code: string): number {
        if (!this.keys.has(code)) return 0;
        return this.pressRamp(this.arrowPressTimestamps[code] ?? 0, 0.5, 1.0, 1.0);
    }

    private applyDeadzone(value: number): number {
        if (Math.abs(value) < this.DEADZONE) {
            return 0;
        }
        // Optional: Re-map 0.1->1.0 to 0.0->1.0 so you don't "jump" when leaving deadzone
        // But for simple testing, just zeroing it is fine.
        return value;
    }

    // Set an object to use as the orbit center for control inputs
    public setOrbitCenter(orbitObj: THREE.Object3D) {
        this.orbitObj = orbitObj
        this.setSeatOrbitMode(true);
    }

    public setRobotPosition(x: number, y: number) {
        this.robotPosition = { x, y };
    }

    public setSeatOrbitMode(seatOrbit: boolean) {
        this.seatOrbitMode = seatOrbit;
    }

    /**
     * Main control loop logic. Call this every frame.
     * Returns an array of ControlItems to be sent over the network.
     */
    public checkInputsAndCreateControlItems(lerobotStatus: nf.common.LerobotStatus): Array<nf.control.ControlItem> {
        // first use keyboard input
        let input = this.getKeyboardState();
        const gpInput = this.getGamepadState();
        const touchInput = this.touchProvider ? this.touchProvider() : null;

        if (touchInput) {
            // Merge touch input the same way we merge gamepad input below.
            input = {
                leftStick: {
                    x: input.leftStick.x + (touchInput.leftStick?.x ?? 0),
                    y: input.leftStick.y + (touchInput.leftStick?.y ?? 0),
                },
                rightStick: {
                    x: input.rightStick.x + (touchInput.rightStick?.x ?? 0),
                    y: input.rightStick.y + (touchInput.rightStick?.y ?? 0),
                },
                buttons: {
                    a: input.buttons.a || (touchInput.buttons?.a ?? false),
                    b: input.buttons.b || (touchInput.buttons?.b ?? false),
                    x: input.buttons.x || (touchInput.buttons?.x ?? false),
                    y: input.buttons.y || (touchInput.buttons?.y ?? false),
                    lb: input.buttons.lb || (touchInput.buttons?.lb ?? false),
                    rb: input.buttons.rb || (touchInput.buttons?.rb ?? false),
                    lt: Math.max(input.buttons.lt, touchInput.buttons?.lt ?? 0),
                    rt: Math.max(input.buttons.rt, touchInput.buttons?.rt ?? 0),
                    start: input.buttons.start || (touchInput.buttons?.start ?? false),
                    select: input.buttons.select || (touchInput.buttons?.select ?? false),
                    lclick: input.buttons.lclick || (touchInput.buttons?.lclick ?? false),
                    rclick: input.buttons.rclick || (touchInput.buttons?.rclick ?? false),
                    dpadUp: input.buttons.dpadUp || (touchInput.buttons?.dpadUp ?? false),
                    dpadDown: input.buttons.dpadDown || (touchInput.buttons?.dpadDown ?? false),
                    dpadLeft: input.buttons.dpadLeft || (touchInput.buttons?.dpadLeft ?? false),
                    dpadRight: input.buttons.dpadRight || (touchInput.buttons?.dpadRight ?? false),
                },
            };
        }

        if (gpInput) {
            // gamepad unlock check  - trigger buttons may return invalid values until pressed.
            if (!this.seenValidTriggers){
                if ((gpInput.buttons.rt > 0 && gpInput.buttons.rt < 1) && (gpInput.buttons.lt > 0 && gpInput.buttons.lt < 1)) {
                    this.seenValidTriggers = true;
                    // Gamepad has been unlocked. Clear the explanatory message
                    const container = document.getElementById('how-to');
                    if (container) {
                        container.textContent = "";
                    }
                } else if (gpInput.buttons.rt == 1 && gpInput.buttons.lt == 1) {
                    // your triggers are not analog. Enable a timed ramp behavior for the triggers
                    this.seenValidTriggers = true;
                    this.useRampForTriggers = true;
                    const container = document.getElementById('how-to');
                    if (container) {
                        container.textContent = "Discrete triggers detected. Using timed ramp."
                    }
                }
            } else {
                // have gp input and it's unlocked. merge it with keyboard input
                input = {
                    leftStick: {
                        x: input.leftStick.x + (gpInput?.leftStick.x ?? 0),
                        y: input.leftStick.y + (gpInput?.leftStick.y ?? 0)
                    },
                    rightStick: {
                        x: input.rightStick.x + (gpInput?.rightStick.x ?? 0),
                        y: input.rightStick.y + (gpInput?.rightStick.y ?? 0)
                    },
                    buttons: {
                        a: input.buttons.a || (gpInput?.buttons.a ?? false),
                        b: input.buttons.b || (gpInput?.buttons.b ?? false),
                        x: input.buttons.x || (gpInput?.buttons.x ?? false),
                        y: input.buttons.y || (gpInput?.buttons.y ?? false),
                        lb: input.buttons.lb || (gpInput?.buttons.lb ?? false),
                        rb: input.buttons.rb || (gpInput?.buttons.rb ?? false),
                        lt: Math.max(input.buttons.lt, gpInput?.buttons.lt ?? 0),
                        rt: Math.max(input.buttons.rt, gpInput?.buttons.rt ?? 0),
                        start: input.buttons.start || (gpInput?.buttons.start ?? false),
                        select: input.buttons.select || (gpInput?.buttons.select ?? false),
                        lclick: input.buttons.lclick || (gpInput?.buttons.lclick ?? false),
                        rclick: input.buttons.rclick || (gpInput?.buttons.rclick ?? false),
                        dpadUp: input.buttons.dpadUp || (gpInput?.buttons.dpadUp ?? false),
                        dpadDown: input.buttons.dpadDown || (gpInput?.buttons.dpadDown ?? false),
                        dpadLeft: input.buttons.dpadLeft || (gpInput?.buttons.dpadLeft ?? false),
                        dpadRight: input.buttons.dpadRight || (gpInput?.buttons.dpadRight ?? false),
                    }
                };
            }
        }
        
        const messages: nf.control.ControlItem[] = [];
        const now = Date.now() / 1000;

        // Calculate Vector (Left Stick + Triggers)
        // If left unchanged, this is the axis aligned movement perspective
        // Since the gripper's camera is stabilized and reprojected in the room's frame of reference,
        // it is also a motion vector that aligns with the gripper's perspective.
        let netTrigger = input.buttons.rt - input.buttons.lt;
        if (this.useRampForTriggers && netTrigger !== 0) {
            this.rampedTriggerEMA = this.rampedTriggerEMA * 0.99 + netTrigger * 0.01
            netTrigger = this.rampedTriggerEMA
        }
        let vx = input.leftStick.x;
        let vy = input.leftStick.y;
        let vz = netTrigger;

        // in orbit mode, lateral movements orbit a given position and
        // forward/back movements move away/towards it.
        if (this.seatOrbitMode && this.orbitObj && this.robotPosition) {

            // get position of orbitObj in x,y plane in robot space
            const p = new THREE.Vector3();
            this.orbitObj.getWorldPosition(p);
            const orbitX = p.x
            const orbitY = -p.z;

            // Vector from orbit position to robot
            const dx = this.robotPosition.x - orbitX;
            const dy = this.robotPosition.y - orbitY;
            const dist = Math.sqrt(dx * dx + dy * dy);

            // We can only orbit if we aren't at the exact center.
            if (dist > 1e-6) {
                const rx = dx / dist; // Radial X
                const ry = dy / dist; // Radial Y

                // Tangential Unit Vector (Clockwise Perpendicular: [y, -x])
                const tx = ry; 
                const ty = -rx;

                // Transform Stick Inputs: 
                // Stick X -> Tangential (Orbit)
                // Stick Y -> Radial (In/Out)
                const orbitVx = (vx * tx) + (vy * rx);
                const orbitVy = (vx * ty) + (vy * ry);

                vx = orbitVx;
                vy = orbitVy;
            }
        }

        // Normalize and Calculate Speed
        // (x,y,z) is the direction vector, speed is scalar
        const mag = Math.sqrt(vx*vx + vy*vy + vz*vz);
        let speed = 0;
        
        if (mag > 0) {
            vx /= mag;
            vy /= mag;
            vz /= mag;
            // Scale speed (max 0.35 m/s)
            speed = 0.45 * mag;
        }

        // Finger Control A/B
        // Move slow at first, then faster.
        // When button is released, send zero speed.
        let fingerChange = 0;
        if (input.buttons.a) {
            if (!this.aWasHeld) this.aPressTimestamp = Date.now() / 1000;
            fingerChange = this.pressRamp(this.aPressTimestamp, this.GAMEPAD_GRIP_SLOW, this.GAMEPAD_GRIP_FAST, 0.5);
        } else if (input.buttons.b) {
            if (!this.bWasHeld) this.bPressTimestamp = Date.now() / 1000;
            fingerChange = -this.pressRamp(this.bPressTimestamp, this.GAMEPAD_GRIP_SLOW, this.GAMEPAD_GRIP_FAST, 0.5);
        }
        this.aWasHeld = input.buttons.a;
        this.bWasHeld = input.buttons.b;

        // Right stick Y also drives the fingers (up = open) when A/B aren't held.
        if (fingerChange === 0) {
            fingerChange = -input.rightStick.y * this.GAMEPAD_GRIP_FAST;
        }

        // Wrist Control (right stick X)
        const wristChange = input.rightStick.x * this.GAMEPAD_WRIST_DEG_PER_SEC;

        // Rising Edge Detectors for Events

        // Start Button -> Episode Start/Stop
        if (input.buttons.start && !this.startWasHeld) {
            if (lerobotStatus === nf.common.LerobotStatus.LEROBOTSTATUS_RECORDING || lerobotStatus === nf.common.LerobotStatus.LEROBOTSTATUS_EVAL_ACTIVE) {
                Say(`End episode`);
                messages.push(nf.control.ControlItem.create({
                    episodeControl: { command: nf.common.EpCommand.EPCOMMAND_EVAL_STOP }
                }));
            } else if (lerobotStatus === nf.common.LerobotStatus.LEROBOTSTATUS_REC_READY || lerobotStatus === nf.common.LerobotStatus.LEROBOTSTATUS_EVAL_IDLE) {
                messages.push(nf.control.ControlItem.create({
                    episodeControl: { command: nf.common.EpCommand.EPCOMMAND_EVAL_START }
                }));
            }
        }
        this.startWasHeld = input.buttons.start;

        // Y Button -> Set Prompt
        if (input.buttons.y && !this.yWasHeld) {
            this.onSetPrompt();
        }
        this.yWasHeld = input.buttons.y;

        // D-Pad Up -> Tighten Lines
        if (input.buttons.dpadUp && !this.dpadUpWasHeld) {
            console.log('Gamepad: Tension Lines');
            messages.push(nf.control.ControlItem.create({
                command: { name: nf.control.Command.COMMAND_TIGHTEN_LINES }
            }));
        }
        this.dpadUpWasHeld = input.buttons.dpadUp;

        // D-Pad Left -> Half Cal
        if (input.buttons.dpadLeft && !this.dpadLeftWasHeld) {
            messages.push(nf.control.ControlItem.create({
                command: { name: nf.control.Command.COMMAND_HALF_CAL }
            }));
        }
        this.dpadLeftWasHeld = input.buttons.dpadLeft;

        // D-Pad Right -> Grasp
        if (input.buttons.dpadRight && !this.dpadRightWasHeld) {
            messages.push(nf.control.ControlItem.create({
                command: { name: nf.control.Command.COMMAND_GRASP }
            }));
        }
        this.dpadRightWasHeld = input.buttons.dpadRight;

        // nothing is mapped to dpad down at the moment

        // Select/back - stop all.
        // also triggers Lerobot EPCOMMAND_ABANDON
        if (input.buttons.select && !this.selectWasHeld) {
            messages.push(nf.control.ControlItem.create({
                command: { name: nf.control.Command.COMMAND_STOP_ALL }
            }));
        }
        this.selectWasHeld = input.buttons.select;

        // Click left stick or press 5
        // Toggle swing cancellation
        if (input.buttons.lclick && !this.lclickWasHeld) {
            this.toggleSwingC();
        }
        this.lclickWasHeld = input.buttons.lclick;

        // Movement Message (Throttled/Changed)

        // Construct current action array for comparison: [vx, vy, vz, speed, finger, wrist]
        // Note: We only care about direction if magnitude > 0
        const currentAction = [vx, vy, vz, speed, fingerChange, wristChange];

        const hasChanged = !this.arraysEqual(currentAction, this.lastAction);
        const isMoving = mag > 1e-3 || Math.abs(wristChange) > 1.0 || Math.abs(fingerChange) > 1.0;
        const timeSinceSend = now - this.lastSendT;

        // Send a movement if: Data changed OR (we are moving AND it's been > 50ms)
        if (hasChanged || (timeSinceSend > 0.05 && isMoving)) {

            messages.push(nf.control.ControlItem.create({
                move: {
                    direction: { x: vx, y: vy, z: vz },
                    speed: speed,
                    fingerSpeed: fingerChange,
                    wristSpeed: wristChange,
                    directionIsInGripperFrame: !this.seatOrbitMode,
                }
            }));

            // Update state
            for(let i=0; i<6; i++) this.lastAction[i] = currentAction[i];
            this.lastSendT = now;
        }

        return messages;
    }

    private arraysEqual(a: number[], b: Float32Array): boolean {
        for (let i = 0; i < a.length; ++i) {
            // Use small epsilon for float comparison
            if (Math.abs(a[i] - b[i]) > 1e-2) return false;
        }
        return true;
    }

}