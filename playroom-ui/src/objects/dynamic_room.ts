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

    // Default State, one per corner source and in the same slot order as the shared corners
    // array in main.ts: 0-1 anchors, 2-3 eyelets. Overwritten by updateCeilingCorner as poses
    // arrive; the placeholder layout only has to be a box.
    this.corners = [
      new THREE.Vector3(2.5, 3, 2.5),   // 0: anchor 0
      new THREE.Vector3(2.5, 3, -2.5),  // 1: anchor 1
      new THREE.Vector3(-2.5, 3, -2.5), // 2: eyelet 2
      new THREE.Vector3(-2.5, 3, 2.5)   // 3: eyelet 3
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

  /**
   * Move one corner of the room. index identifies which of the four pull points this is
   * (0-1 anchors, 2-3 eyelets), matching the shared corners array in main.ts.
   */
  updateCeilingCorner(index: number, x: number, y: number, z: number) {
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

    // The index buffer walks the ceiling vertices around the perimeter, but the corners arrive
    // in topological order (2 anchors, then 2 eyelets) which is a bowtie: anchor 0 and anchor 1
    // are diagonally opposite. Sorting by angle about the centroid recovers the perimeter for
    // any convex quad at any room yaw. Ascending atan2 keeps the winding the index buffer
    // expects, and the cyclic start point does not matter.
    const cx = this.corners.reduce((s, p) => s + p.x, 0) / this.corners.length;
    const cz = this.corners.reduce((s, p) => s + p.z, 0) / this.corners.length;
    const ring = [...this.corners].sort(
      (a, b) => Math.atan2(a.z - cz, a.x - cx) - Math.atan2(b.z - cz, b.x - cx)
    );

    ring.forEach((pt, i) => {
      // Set Ceiling Vertex (Indices 0-3)
      positions.setXYZ(i, pt.x, pt.y, pt.z);
      // Set Floor Vertex (Indices 4-7) -> Projected to Y=0
      positions.setXYZ(i + 4, pt.x, 0, pt.z);
    });

    positions.needsUpdate = true;
    this.geometry.computeVertexNormals();
  }
}