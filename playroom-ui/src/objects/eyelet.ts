import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { nf } from '../generated/proto_bundle.js';
import { DynamicRoom } from './dynamic_room.ts';

export class Eyelet {
    // single copy of the geometry
    private static modelPromise: Promise<GLTF> | null = null;

    private scene: THREE.Scene;
    private root: THREE.Group;
    public ready: Promise<void>;
    public grommet_pos: THREE.Vector3;

    // properties used to update the room walls when eyelet pose changes.
    private room: DynamicRoom;
    private wallCorner: THREE.Object3D | undefined;
    private grommet: THREE.Object3D | undefined;

    constructor(scene: THREE.Scene, room: DynamicRoom) {
        this.scene = scene;
        this.room = room;

        this.root = new THREE.Group();
        this.scene.add(this.root);
        this.grommet_pos = new THREE.Vector3();

        // Kick off loading (or attach to existing loading process)
        this.ready = this.loadSharedModel();
    }

    private async loadSharedModel() {
        if (!Eyelet.modelPromise) {
            const loader = new GLTFLoader();
            Eyelet.modelPromise = loader.loadAsync(import.meta.env.VITE_ASSET_BUCKET_URL+'/assets/models/eyelet.glb');
        }
        
        try {
            const masterGltf = await Eyelet.modelPromise;
            const clonedScene = masterGltf.scene.clone();
            this.root.add(clonedScene);
            
            this.wallCorner = clonedScene.getObjectByName('wall_corner');
            this.grommet = clonedScene.getObjectByName('grommet');

            if (this.wallCorner) {
                this.updateRoomCorner();
            }

            if (this.grommet) {
                this.grommet.getWorldPosition(this.grommet_pos);
            }
        } catch (error) {
            console.error('Error loading eyelet.glb:', error);
            // Fallback visualization
            const mesh = new THREE.Mesh( 
                new THREE.BoxGeometry(0.1, 0.1, 0.1), 
                new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true })
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
     * Set the initial pose (including rotation) identical to the anchor setup 
     * so that the eyelet perfectly aligns with the room corner geometry.
     */
    public setPose(pose: nf.common.IPose) {
        if (pose.position && pose.rotation) {
            this.root.position.set(
                (pose.position.x ?? 0),
                (pose.position.z ?? 0),
                -(pose.position.y ?? 0)
            );

            const rx = (pose.rotation.x ?? 0);
            const ry = (pose.rotation.z ?? 0); 
            const rz = -(pose.rotation.y ?? 0);
            
            const theta = Math.sqrt(rx * rx + ry * ry + rz * rz);
            if (theta > 0.0001) {
                const axis = new THREE.Vector3(rx / theta, ry / theta, rz / theta);
                this.root.quaternion.setFromAxisAngle(axis, theta);
            } else {
                this.root.quaternion.identity();
            }

            this.updateRoomCorner();
            if (this.grommet){
                this.grommet.getWorldPosition(this.grommet_pos);
            } else {
                this.grommet_pos.copy(this.root.position);
            }
        }
    }

    /**
     * Eyelets don't have known rotations, just positions.
     */
    public setPosition(pos: nf.common.IVec3) {
        this.root.position.set(
            (pos.x ?? 0),
            (pos.z ?? 0),
            -(pos.y ?? 0)
        );

        this.updateRoomCorner();
        if (this.grommet) {
            this.grommet.getWorldPosition(this.grommet_pos);
        } else {
            this.grommet_pos.copy(this.root.position);
        }
    }

    /** Cleanly wipe this eyelet from the scene */
    public dispose() {
        this.scene.remove(this.root);
    }
}