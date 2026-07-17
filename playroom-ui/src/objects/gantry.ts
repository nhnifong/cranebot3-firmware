import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { nf } from '../generated/proto_bundle.js';
import { mapRange } from '../utils.ts'

const SM_PILOT_MAX_SP = 0.25;

export class Gantry {
    // single copy of the geometry
    private static modelPromise: Promise<GLTF> | null = null;

    private scene: THREE.Scene;
    private root: THREE.Group;
    public ready: Promise<void>;
    public position: THREE.Vector3;

    // sub objects
    private velPoint: THREE.Object3D | undefined;
    private arrowHead: THREE.Object3D | undefined;
    private arrowShaft: THREE.Object3D | undefined;

    // constants derived from model
    private arrowHeadMinY: number;
    private arrowHeadMaxY: number;

    constructor(scene: THREE.Scene) {
        this.scene = scene;

        this.root = new THREE.Group();
        this.scene.add(this.root);

        this.root.position.set(0, 1.5, 0 );
        this.position = this.root.position;

        // Kick off loading (or attach to existing loading process)
        this.ready = this.loadSharedModel();

        this.arrowHeadMinY = 0.086760;
        this.arrowHeadMaxY = 0.365824;
    }

    private async loadSharedModel() {
        if (!Gantry.modelPromise) {
            const loader = new GLTFLoader();
            Gantry.modelPromise = loader.loadAsync(import.meta.env.VITE_ASSET_BUCKET_URL+'/assets/models/gantry.glb');
        }
        
        try {
            const masterGltf = await Gantry.modelPromise;
            const clonedScene = masterGltf.scene.clone();
            this.root.add(clonedScene);

            // Find sub-objects in clone
            this.velPoint = clonedScene.getObjectByName('vel_point');
            this.arrowHead = clonedScene.getObjectByName('arrowhead'); // child of velPoint
            this.arrowShaft = clonedScene.getObjectByName('shaft'); // child of velPoint

            // In the model, when  the shaft is at scale=1, that's the longest it's suppoed to be.
            // The initial position of the arrow head in the  model is the place it should be at maximum extent.
            // the initial location of the shaft in the model is it's base, and that's where the head should be at it's minimum extent.
            this.arrowHeadMaxY = this.arrowHead!.position.y;
            this.arrowHeadMinY = this.arrowShaft!.position.y;

            this.velPoint!.visible = true;

        } catch (error) {
            console.error('Error loading gantry.glb:', error);
            // Fallback visualization
            const mesh = new THREE.Mesh( 
                new THREE.BoxGeometry(0.1, 0.1, 0.1), 
                new THREE.MeshBasicMaterial({ color: 0xff0000, wireframe: true })
            );
            this.root.add(mesh);
        }
    }

    public setPosition(position: nf.common.IVec3) {
        // swap y and z
        this.root.position.set(
            (position.x ?? 0),
            (position.z ?? 0),
            -(position.y ?? 0)
        );
    }

public setVelocity(velocity: nf.common.IVec3) {
        if (!this.velPoint || !this.arrowHead || !this.arrowShaft) {
            return;
        }

        const x = velocity.x ?? 0;
        const y = velocity.y ?? 0;
        const z = velocity.z ?? 0;

        // Convert to ThreeJS Space
        const dir = new THREE.Vector3(x, z, -y);

        // we are given the actual commanded velocity in meters per second.
        // Normaize this based on the maximum movement speed of a Stringman Pilot.
        const length = dir.length() / SM_PILOT_MAX_SP;

        // Visibility Check
        if (length < 0.001) {
            this.velPoint.visible = false;
            return;
        }
        this.velPoint.visible = true;

        // Orientation (Rotate velPoint)
        // This rotates both shaft and head together
        dir.normalize();
        this.velPoint.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0), // because the arrow points up in blender
            dir
        );
        
        // Shaft: Scale Y to match length
        // In blender the shaft's lenght is it's maximum length.
        this.arrowShaft.scale.set(1, length, 1);

        // Head: Move Y to sit on top of the shaft
        this.arrowHead.position.set(0, mapRange(length, 0, 1, this.arrowHeadMinY, this.arrowHeadMaxY), 0);
    }
}