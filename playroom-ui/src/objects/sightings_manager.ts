import * as THREE from 'three';
import { nf } from '../generated/proto_bundle.js';

export class SightingsManager {
    private scene: THREE.Scene;
    private root: THREE.Group;
    
    // Store active dots
    private dots: { mesh: THREE.Mesh; birthTime: number }[] = [];
    
    // Configuration
    private readonly MAX_AGE = 5.0; // Seconds to live
    private color: number;
    private radius: number;
    
    // Shared geometry for performance
    private sharedGeometry: THREE.SphereGeometry;

    constructor(scene: THREE.Scene, options: { color?: number; radius?: number } = {}) {
        this.scene = scene;
        this.color = options.color ?? 0xFFFFFF;
        this.radius = options.radius ?? 0.01;

        this.root = new THREE.Group();
        this.scene.add(this.root);

        // Low poly sphere is fine for small dots
        this.sharedGeometry = new THREE.SphereGeometry(this.radius, 8, 8);
    }

    public update() {
        const now = Date.now() / 1000;

        for (let i = this.dots.length - 1; i >= 0; i--) {
            const dot = this.dots[i];
            const age = now - dot.birthTime;

            if (age >= this.MAX_AGE) {
                this.root.remove(dot.mesh);
                // Cast to Material for disposal to free GPU resources
                (dot.mesh.material as THREE.Material).dispose();
                this.dots.splice(i, 1);
            } else {
                // Opacity goes from 1.0 down to 0.0
                const opacity = 1.0 - (age / this.MAX_AGE);
                (dot.mesh.material as THREE.MeshBasicMaterial).opacity = opacity;
            }
        }
    }

    public addSingleItem(x: number, y: number, z: number) {
        const material = new THREE.MeshBasicMaterial({
            color: this.color,
            transparent: true,
            opacity: 1.0,
            depthWrite: false // Prevents transparency sorting artifacts/flickering
        });

        const mesh = new THREE.Mesh(this.sharedGeometry, material);
        
        // Coordinate Conversion: Z-up (Robot) -> Y-up (Three.js)
        mesh.position.set(x, y, z);

        this.root.add(mesh);
        this.dots.push({ 
            mesh, 
            birthTime: Date.now() / 1000 
        });
    }

    public handleSightings(data: nf.telemetry.IGantrySightings) {
        if (!data.sightings) return;

        data.sightings.forEach(vec => {
            this.addSingleItem(
                vec.x ?? 0, 
                vec.z ?? 0,
                -(vec.y ?? 0), 
            );
        });
    }
}
