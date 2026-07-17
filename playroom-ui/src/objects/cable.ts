import * as THREE from 'three';

export class Cable {
    private mesh: THREE.Line;
    private geometry: THREE.BufferGeometry;
    private curve: THREE.QuadraticBezierCurve3;

    // Define resolution
    private divisions: number;

    constructor(scene: THREE.Scene, color: number = 0x333333) {
        // Setup reusable objects
        this.curve = new THREE.QuadraticBezierCurve3();
        this.geometry = new THREE.BufferGeometry();
        this.divisions = 20
        
        // Material for a thin line
        const material = new THREE.LineBasicMaterial({ 
            color: color,
            linewidth: 2
        });

        // Create the mesh
        this.mesh = new THREE.Line(this.geometry, material);
        
        // re-fill the buffer with the correct number of dummy points (21)
        const zeros = new Array(this.divisions + 1).fill(new THREE.Vector3(0, 0, 0));
        this.geometry.setFromPoints(zeros);
        
        scene.add(this.mesh);
    }

    /**
     * Updates the cable geometry.
     * @param start World position of start point
     * @param end World position of end point
     * @param sag Amount of vertical drop at the lowest point (in meters)
     */
    update(start: THREE.Vector3, end: THREE.Vector3, sag: number) {
        // Calculate the Control Point
        // The midpoint of the straight line connecting the two points
        const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
        
        // Apply sag downwards (Three.js Y is up, so we subtract Y)
        // We multiply sag by 2 because the Bezier control point pulls the curve
        // exactly halfway to itself. To get a visible sag of X, the control point needs to be at 2X.
        mid.y -= (sag * 2);

        // Update the Curve Math
        this.curve.v0.copy(start);
        this.curve.v1.copy(mid); // The "Control Point" pulls the line down
        this.curve.v2.copy(end);

        // Sample points along the curve
        const points = this.curve.getPoints(this.divisions);

        // Update the Geometry buffer
        this.geometry.setFromPoints(points);
    }
}