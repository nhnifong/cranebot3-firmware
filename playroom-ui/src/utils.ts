import * as THREE from 'three';

export interface CameraIntrinsics {
    fx: number;
    fy: number;
    cx: number;
    cy: number;
}

export interface DistortionCoefficients {
    k1: number;
    k2: number;
    p1: number;
    p2: number;
    k3?: number;
}

export interface ImageShape {
    width: number;
    height: number;
}

export interface CameraCalibration {
    K: CameraIntrinsics,
    D: DistortionCoefficients,
    imageShape: ImageShape // the shape of the image that the K matrix was created from.
}

const rpiCam3Cal: CameraCalibration = {
    K: { fx: 1424, fy: 1424, cx: 960.0, cy: 540.0 },
    D: { k1: 0.011584, k2: 0.187238, p1: -0.001261, p2: 0.000583, k3: -0.388072 },
    imageShape: { width: 1920, height: 1080 }
};

export const TargetColors = {
    'mouse':     '#00FF00', // green
    'selected':  '#FFA500', // orange
    'hovered':   '#00FFFF', // cyan
    'seen':      '#FFFFFF', // white
    'movingTo': '#007fff', // blue
    'grasped':   '#ffd700', // gold
};

/**
 * Equivalent to cv2.undistortPoints()
 * Uses an iterative method to reverse the Brown-Conrady distortion model.
 * Returns undistorted pixel coordinates (Origin top-left).
 */
export function undistortPoint(
    point: THREE.Vector2,
    K: CameraIntrinsics,
    D: DistortionCoefficients,
    iterations: number = 5
): THREE.Vector2 {
    // Normalize (Pixel -> Normalized Image Plane)
    const x0 = (point.x - K.cx) / K.fx;
    const y0 = (point.y - K.cy) / K.fy;

    let x = x0;
    let y = y0;

    // Iteratively solve for (x, y) ideal
    for (let i = 0; i < iterations; i++) {
        const r2 = x * x + y * y;
        const r4 = r2 * r2;
        const r6 = r2 * r4;

        const kRadial = 1 + (D.k1 * r2) + (D.k2 * r4) + ((D.k3 ?? 0) * r6);
        const deltaX = (2 * D.p1 * x * y) + (D.p2 * (r2 + 2 * x * x));
        const deltaY = (D.p1 * (r2 + 2 * y * y)) + (2 * D.p2 * x * y);

        x = (x0 - deltaX) / kRadial;
        y = (y0 - deltaY) / kRadial;
    }

    // Denormalize (Normalized -> Pixel)
    return new THREE.Vector2(
        x * K.fx + K.cx,
        y * K.fy + K.cy
    );
}

/**
 * Batch project normalized [0,1] UV coordinates (Origin Top-Left) from a Three.js Camera's 
 * point of view to the floor plane (Y=0) using the rpiCam3Cal intrinsics.
 * Returns null if the ray does not intersect the floor or is invalid.
 */
export function projectPixelsToFloor(
    normalizedPixels: THREE.Vector2[],
    camera: THREE.PerspectiveCamera
): (THREE.Vector3 | null)[] {
    const { K, D, imageShape } = rpiCam3Cal;
    const results: (THREE.Vector3 | null)[] = [];
    
    // Ensure the camera's world matrix is up to date
    camera.updateMatrixWorld();
    const cameraWorldPos = new THREE.Vector3();
    camera.getWorldPosition(cameraWorldPos);

    // Reuse vector objects to reduce garbage collection
    const rayDir = new THREE.Vector3();

    for (const p of normalizedPixels) {
        // Scale normalized [0,1] to actual pixel coordinates
        const px = p.x * imageShape.width;
        const py = p.y * imageShape.height;

        // Undistort the point (returns undistorted PIXEL coords)
        const undistortedPx = undistortPoint(new THREE.Vector2(px, py), K, D);

        // Convert to Normalized Image Plane (u, v) in Camera Frame
        // Map from Image Space (Y-Down) to Three.js Camera Space (Y-Up, -Z Forward)
        
        const u = (undistortedPx.x - K.cx) / K.fx;
        const v = (undistortedPx.y - K.cy) / K.fy;

        // Construct Ray Direction in Camera Local Space
        // u is Right (+X), v is Down (so -v is Up/+Y), -1 is Forward (-Z)
        rayDir.set(u, -v, -1).normalize();

        // Transform Direction to World Space
        // transformDirection applies only rotation (ignores translation)
        rayDir.transformDirection(camera.matrixWorld);

        // Intersect ray with the floor plane (Y=0)
        // Check for parallel rays
        if (Math.abs(rayDir.y) < 1e-6) {
            results.push(null);
            continue;
        }

        const s = -cameraWorldPos.y / rayDir.y;

        // Ensure intersection is in front of the ray (positive s)
        if (s > 0) {
            results.push(new THREE.Vector3(
                cameraWorldPos.x + s * rayDir.x,
                0,
                cameraWorldPos.z + s * rayDir.z
            ));
        } else {
            results.push(null);
        }
    }

    return results;
}

/**
 * Project world coordinates on the floor (Y=0) back to normalized UV coordinates [0,1] (Origin Top-Left)
 * using the rpiCam3Cal intrinsics.
 * Returns null if the point is behind the camera.
 */
export function projectFloorToPixels(
    floorPoints: THREE.Vector3[],
    camera: THREE.PerspectiveCamera,
    finalScale: THREE.Vector2,
): (THREE.Vector2 | null)[] {
    const { K, D, imageShape } = rpiCam3Cal;
    const results: (THREE.Vector2 | null)[] = [];

    camera.updateMatrixWorld();
    const invMatrix = camera.matrixWorldInverse;

    const localPos = new THREE.Vector3();

    for (const fp of floorPoints) {
        // Transform World Point to Camera Local Space
        localPos.copy(fp).applyMatrix4(invMatrix);

        // Convert Three.js Local Space to Image Space logic
        // Three Local: Forward is -Z, Up is +Y
        // Image Space: Forward is +Z, Down is +Y
        const imgX = localPos.x;
        const imgY = -localPos.y;
        const imgZ = -localPos.z;

        // Check if point is behind camera (Z <= 0 in image frame)
        if (imgZ <= 0) {
            results.push(null);
            continue;
        }

        // Project to Normalized Image Plane (Perspective Divide)
        const xn = imgX / imgZ;
        const yn = imgY / imgZ;

        // Apply Distortion (Forward Model)
        const distorted = distortNormalizedPoint(xn, yn, D);

        // Apply Intrinsics (Normalized -> Pixel)
        const u = distorted.x * K.fx + K.cx;
        const v = distorted.y * K.fy + K.cy;

        // Normalize to [0, 1]
        results.push(new THREE.Vector2(
            u / imageShape.width * finalScale.x,
            v / imageShape.height * finalScale.y
        ));
    }

    return results;
}

function distortNormalizedPoint(x: number, y: number, D: DistortionCoefficients): THREE.Vector2 {
    const r2 = x*x + y*y;
    const r4 = r2*r2;
    const r6 = r2*r4;

    const kRadial = 1 + D.k1*r2 + D.k2*r4 + (D.k3 ?? 0)*r6;
    const dx = 2*D.p1*x*y + D.p2*(r2 + 2*x*x);
    const dy = D.p1*(r2 + 2*y*y) + 2*D.p2*x*y;

    return new THREE.Vector2(
        x*kRadial + dx,
        y*kRadial + dy
    );
}

export function mapRange(
    value: number,
    inputMin: number,
    inputMax: number,
    outputMin: number,
    outputMax: number
): number {
    const inputRange = inputMax - inputMin;
    const outputRange = outputMax - outputMin;

    if (inputRange === 0) {
        // Avoid division by zero
        return (outputMin + outputMax) / 2;
    }

    const scaledValue = (value - inputMin) * outputRange / inputRange;
    return outputMin + scaledValue;
}

export function Say(text: string): void {
  const utterance = new SpeechSynthesisUtterance(text);

  utterance.rate = 1.0; // Speed (0.1 to 10)
  utterance.pitch = 1.0; // Pitch (0 to 2)
  utterance.volume = 1.0; // Volume (0 to 1)

  // 3. Tell the browser to speak
  window.speechSynthesis.speak(utterance);
}

export function Listen(): Promise<string> {
  return new Promise((resolve, reject) => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      reject('Speech recognition not supported in this browser');
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event: any) => {
      resolve(event.results[0][0].transcript as string);
    };
    recognition.onerror = (event: any) => {
      reject(event.error as string);
    };
    recognition.start();
  });
}