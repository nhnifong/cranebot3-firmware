import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { nf } from '../generated/proto_bundle.js';

export class DynamicRoom {
  private static modelPromise: Promise<GLTF> | null = null;

  private scene: THREE.Scene;
  private root: THREE.Group;
  public ready: Promise<void>;

  // point representing perspective of user holding gamepad
  public userPers: THREE.Object3D | undefined;
  private hamper: THREE.Object3D | undefined;
  private reticule: THREE.Object3D | undefined;
  private toybox: THREE.Object3D | undefined;
  private trash_can: THREE.Object3D | undefined;

  // moving walls
  mesh: THREE.Mesh;
  private geometry: THREE.BufferGeometry;
  private corners: THREE.Vector3[]; 

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.root = new THREE.Group();
    this.scene.add(this.root);

    // Kick off loading (or attach to existing loading process)
    this.ready = this.loadSharedModel();

    // Define Default State (4 Ceiling Corners)
    this.corners = [
      new THREE.Vector3(2.5, 3, -2.5), // 0: Front Right
      new THREE.Vector3(2.5, 3, 2.5),  // 1: Back Right
      new THREE.Vector3(-2.5, 3, 2.5), // 2: Back Left
      new THREE.Vector3(-2.5, 3, -2.5) // 3: Front Left
    ];

    // Setup BufferGeometry
    this.geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array(8 * 3); // 8 points (4 ceil + 4 floor)
    this.geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

    // Indices for looking continuously at BackSide (Internal View)
    this.geometry.setIndex([
      0, 4, 5,  0, 5, 1, // Front Wall
      1, 5, 6,  1, 6, 2, // Right Wall
      2, 6, 7,  2, 7, 3, // Back Wall
      3, 7, 4,  3, 4, 0, // Left Wall
      0, 1, 2,  0, 2, 3, // Ceiling
      5, 4, 7,  5, 7, 6  // Floor
    ]);

    // Create Mesh
    const material = new THREE.MeshStandardMaterial({
      color: 0xFFFDD1, // cream
      side: THREE.FrontSide, // already did this via winding order above
      flatShading: true // Essential for dynamic non-planar quads
    });

    this.mesh = new THREE.Mesh(this.geometry, material);
    this.root.add(this.mesh);

    // Initial Draw
    this.updateGeometry();
  }

  private async loadSharedModel() {
      if (!DynamicRoom.modelPromise) {
          const loader = new GLTFLoader();
          DynamicRoom.modelPromise = loader.loadAsync((import.meta.env.VITE_ASSET_BUCKET_URL ?? '')+'/assets/playroom/models/decor.glb');
      }
      
      try {
          const masterGltf = await DynamicRoom.modelPromise;
          const clonedScene = masterGltf.scene.clone();
          this.root.add(clonedScene);

          // Find sub-objects in clone
          this.userPers = clonedScene.getObjectByName('user_pers');
          this.hamper = clonedScene.getObjectByName('hamper_tag');
          this.reticule = clonedScene.getObjectByName('reticule');
          this.toybox = clonedScene.getObjectByName('toybox');
          this.trash_can = clonedScene.getObjectByName('trash_can');

      } catch (error) {
          console.error('Error loading decor.glb:', error);
      }
  }

  updateCeilingCorner(x: number, y: number, z: number) {
    let index = -1;

    // Determine index based on Quadrants
    if (x > 0) {
      // If X is Positive: Z negative is index 0, Z positive is index 1
      index = z < 0 ? 0 : 1;
    } else {
      // If X is Negative: Z positive is index 2, Z negative is index 3
      index = z > 0 ? 2 : 3;
    }

    if (this.corners[index]) {
      this.corners[index].set(x, y, z);
      this.updateGeometry();
    }
  }

  public setNamedObjectPosition(name: string, position: nf.common.IVec3) {
    const target = name === 'hamper' ? this.hamper
                 : name === 'toys' ? this.toybox
                 : name === 'trash' ? this.trash_can
                 : name === 'gamepad' ? this.userPers
                 : undefined;
    if (target) {
      target.position.set(
          (position.x ?? 0),
          (position.z ?? 0),
          -(position.y ?? 0)
      );
    }
  }

  public setReticule(position: THREE.Vector3 | null) {
    if (this.reticule) {
      if (position) {
        this.reticule.position.set(
          position.x,
          this.reticule.position.y, // don't change y
          position.z
        );
        this.reticule.visible == true;
      } else {
        // position==null means the reticule needs to be hidden
        this.reticule.visible == false;
      }
    }
  }

  private updateGeometry() {
    const positions = this.geometry.attributes.position;

    this.corners.forEach((pt, i) => {
      // Set Ceiling Vertex (Indices 0-3)
      positions.setXYZ(i, pt.x, pt.y, pt.z);
      // Set Floor Vertex (Indices 4-7) -> Projected to Y=0
      positions.setXYZ(i + 4, pt.x, 0, pt.z);
    });

    positions.needsUpdate = true;
    this.geometry.computeVertexNormals();
  }
}