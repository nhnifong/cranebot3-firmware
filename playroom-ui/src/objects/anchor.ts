import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { nf } from '../generated/proto_bundle.js';
import { DynamicRoom } from './dynamic_room.ts';

export class Anchor {
    // single copy of the geometry
    private static modelPromise: Promise<GLTF> | null = null;

    private scene: THREE.Scene;
    private root: THREE.Group;
    public ready: Promise<void>;
    public grommet_pos: THREE.Vector3;

    // properties used to update the room walls when anchor pose changes.
    private room: DynamicRoom;
    private wallCorner: THREE.Object3D | undefined;

    private grommet: THREE.Object3D | undefined;
    public camera: THREE.PerspectiveCamera | undefined;

    constructor(scene: THREE.Scene, room: DynamicRoom) {
        this.scene = scene;
        this.room = room;

        this.root = new THREE.Group();
        this.scene.add(this.root);
        this.grommet_pos = new THREE.Vector3();

        // Kick off loading (or attach to existing loading process)
        this.ready = this.loadSharedModel();
    }

    public getInteractableMesh(): THREE.Object3D | undefined {
        return this.root.getObjectByName('anchor');
    }

    private async loadSharedModel() {
        // if this is the first anchor, download the asset
        if (!Anchor.modelPromise) {
            const loader = new GLTFLoader();
            Anchor.modelPromise = loader.loadAsync(import.meta.env.VITE_ASSET_BUCKET_URL+'/assets/models/anchor.glb');
        }
        
        try {
            // Wait for the download to finish
            const masterGltf = await Anchor.modelPromise;
            // Clone the scene.
            const clonedScene = masterGltf.scene.clone();
            this.root.add(clonedScene);
            // Find sub-objects in clone
            this.wallCorner = clonedScene.getObjectByName('wall_corner');
            this.grommet = clonedScene.getObjectByName('grommet');
            const anchorCam = clonedScene.getObjectByName('camera');

            if (this.wallCorner) {
                // Initialize position immediately if needed
                this.updateRoomCorner();
            }

            if (this.grommet) {
                this.grommet.getWorldPosition(this.grommet_pos);
            }

            // initialize a model of the raspi camera module 3 in the anchor for raycasting targets
            if (anchorCam && (anchorCam as THREE.PerspectiveCamera).isPerspectiveCamera) {
                this.camera = anchorCam as THREE.PerspectiveCamera;
                this.camera.aspect = 16/9;
                this.camera.updateProjectionMatrix();
            }
        } catch (error) {
            console.error('Error loading anchor.glb:', error);
            // Fallback visualization
            const mesh = new THREE.Mesh( 
                new THREE.BoxGeometry(0.1, 0.1, 0.1), 
                new THREE.MeshBasicMaterial({ color: 0xff0000, wireframe: true })
            );
            this.root.add(mesh);
        }
    }

    private updateRoomCorner() {
        if (!this.wallCorner) return;
        this.root.updateMatrixWorld(true);
        const pos = new THREE.Vector3();
        this.wallCorner.getWorldPosition(pos);
        this.room.updateCeilingCorner(pos.x, pos.y, pos.z);
    }

    /**
     * Updates the pose of the anchor relative to the room.
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

            // Update the Room Corner
            this.updateRoomCorner();
            if (this.grommet){
                this.grommet.getWorldPosition(this.grommet_pos);
            } else {
                this.grommet_pos = this.root.position.clone();
            }
        }
    }

    /** Cleanly wipe this anchor from the scene */
    public dispose() {
        this.scene.remove(this.root);
    }
}