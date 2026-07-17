import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { nf } from '../generated/proto_bundle.js';
import { SightingsManager } from '../objects/sightings_manager.ts'

export class Gripper {
    // single copy of the geometry
    private static modelPromise: Promise<GLTF> | null = null;

    private scene: THREE.Scene;
    private root: THREE.Group;
    public ready: Promise<void>;
    public grommet_pos: THREE.Vector3;
    private grommet: THREE.Object3D | undefined;
    private tempVec = new THREE.Vector3();

    // Animation Properties
    private mixer: THREE.AnimationMixer | undefined;
    private action: THREE.AnimationAction | undefined;
    private clipDuration: number = 0;


    constructor(scene: THREE.Scene) {
        this.scene = scene;
        this.root = new THREE.Group();
        this.scene.add(this.root);
        this.root.position.set(0, 1, 0 );
        this.grommet_pos = this.root.position.clone();

        this.ready = this.loadSharedModel();
    }

    public getInteractableMesh(): THREE.Object3D | undefined {
        return this.root.getObjectByName('gripper');
    }

    private async loadSharedModel() {
        if (!Gripper.modelPromise) {
            const loader = new GLTFLoader();
            Gripper.modelPromise = loader.loadAsync(import.meta.env.VITE_ASSET_BUCKET_URL+'/assets/models/gripper.glb');
        }
        
        try {
            // Wait for the download to finish
            const masterGltf = await Gripper.modelPromise;
            // Clone the scene.
            const clonedScene = masterGltf.scene.clone();
            this.root.add(clonedScene);

            // Find sub-objects in clone
            this.grommet = clonedScene.getObjectByName('grommet');

            if (this.grommet) {
                this.grommet.getWorldPosition(this.grommet_pos);
            }

            // Animation setup
            // Check if the GLB had animations
            if (masterGltf.animations && masterGltf.animations.length > 0) {
                // Create a mixer rooted to this specific clone
                this.mixer = new THREE.AnimationMixer(clonedScene);

                // Use the first clip found in the file
                const clip = masterGltf.animations[0];
                this.clipDuration = clip.duration;

                // Create the action
                this.action = this.mixer.clipAction(clip);
                
                // "Play" it, but pause immediately so we can manual scrub
                this.action.play();
                this.action.paused = true;
                
                // Initialize to a neutral position
                this.setJawState(0.3);
            }
        } catch (error) {
            console.error('Error loading gripper.glb:', error);
            // Fallback visualization
            const mesh = new THREE.Mesh( 
                new THREE.BoxGeometry(0.1, 0.1, 0.1), 
                new THREE.MeshBasicMaterial({ color: 0xff0000, wireframe: true })
                );
            this.root.add(mesh);
        }
    }

    /**
     * Updates the pose of the gripper relative to the room.
     * Pose is given in Z-up space. convert to Y-up
     */
    public setPose(pose: nf.common.IPose) {
        if (pose.position && pose.rotation) {

            // swap y and z
            this.root.position.set(
                (pose.position.x ?? 0),
                (pose.position.z ?? 0),
                -(pose.position.y ?? 0)
                );

            // Rotation (Rodrigues Vector)
            const rx = (pose.rotation.x ?? 0);
            const ry = (pose.rotation.z ?? 0); // swap y z
            const rz = -(pose.rotation.y ?? 0);
            
            // Calculate magnitude (angle)
            const theta = Math.sqrt(rx * rx + ry * ry + rz * rz);
            if (theta > 0.0001) {
                const axis = new THREE.Vector3(rx / theta, ry / theta, rz / theta);
                this.root.quaternion.setFromAxisAngle(axis, theta);
            } else {
                this.root.quaternion.identity();
            }

            // update the world position of sub-objects (empties)
            if (this.grommet){
                this.grommet.getWorldPosition(this.grommet_pos);
            } else {
                this.grommet_pos = this.root.position.clone();
            }
        }
    }

    /**
     * Animate the jaws based on a 0-1 open percentage.
     * 0.0 = Start of animation (Closed)
     * 1.0 = End of animation (Fully Open)
     */
    public setJawState(openPercent: number) {
        if (this.mixer && this.action) {
            // Clamp between 0 and 1
            const safePercent = Math.max(0, Math.min(1, openPercent));

            // Map percentage to the exact timestamp in seconds
            this.action.time = safePercent * this.clipDuration;

            // Force the mixer to update the model immediately 
            // (pass 0 delta time since we set absolute time manually)
            this.mixer.update(0);
        }
    }

    public setSensorValues(data: nf.telemetry.IGripperSensors, laserTrace: SightingsManager) {
        if (data.angle) {
            this.setJawState(1-(data.angle + 90)/180)
        }
        if (data.wrist) {
            
        }
        if (data.range){
            // The laser points down the local negative Y axis of the gripper in Three.js
            // We project a point 'range' meters away into world space.
            this.tempVec.set(0, -data.range, 0);
            this.root.localToWorld(this.tempVec);
            laserTrace.addSingleItem(this.tempVec.x, this.tempVec.y, this.tempVec.z);
        }
    }
}