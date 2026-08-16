import * as THREE from 'three';
import Hls from 'hls.js';
import { nf } from '../generated/proto_bundle.js';
import { TargetColors } from '../utils.ts';
import { TargetListManager } from '../ui/target_list_manager.ts';

// Radius (meters) of the target rings laid over the projected floor image.
const TARGET_RING_OUTER = 0.06;
const TARGET_RING_INNER = 0.045;
// How close a floor point has to be to count as hitting a target. Slightly wider
// than the ring, since the 3D view is often looking at the floor edge-on.
const TARGET_PICK_RADIUS = 0.09;

/**
 * Renders a video stream (feed 3 — orthographic color projection) as a texture
 * on a flat quad lying on the floor, centered at the origin.
 *
 * Coordinate mapping:
 *   image +X  →  Three.js +X
 *   image +Y (up / toward image top)  →  Three.js +Z
 *
 * Achieved by rotating a PlaneGeometry (XY plane, flipY=true texture) by
 * +90° around the X axis, which maps local +Y → world +Z.
 */
export class FloorProjection {
    private video: HTMLVideoElement;
    private img: HTMLImageElement;
    private material: THREE.MeshBasicMaterial;
    private mesh: THREE.Mesh;
    private peerConnection: RTCPeerConnection | null = null;
    private hls: Hls | null = null;
    private isLocalMode = false;

    // Flat rings floating just above the projected image, mirroring the target
    // circles the 2D overlays draw on the anchor cam feeds. Each marker is a ring
    // (status colour) plus a disc underneath it that only shows when the target is
    // hovered or selected. Markers are pooled and hidden rather than destroyed,
    // since the target list churns on every update.
    private targetsRoot: THREE.Group;
    private ringGeometry: THREE.RingGeometry;
    private discGeometry: THREE.CircleGeometry;
    private markerPool: { ring: THREE.Mesh; disc: THREE.Mesh }[] = [];
    private lastTargets: nf.telemetry.IOneTarget[] = [];
    private targetListManager: TargetListManager | null;
    // Green ring previewing the target the "Add target" button would create,
    // the 3D counterpart of the video feeds' green new-item square.
    private provisionalRing: THREE.Mesh;

    constructor(scene: THREE.Scene, sizeMeters: number = 5, yOffset: number = 0.001,
                tlm: TargetListManager | null = null) {
        this.video = document.createElement('video');
        this.video.autoplay = true;
        this.video.muted = true;
        this.video.playsInline = true;
        this.video.style.display = 'none';
        document.body.appendChild(this.video);

        this.img = document.createElement('img');
        this.img.crossOrigin = 'anonymous';
        this.img.style.display = 'none';
        document.body.appendChild(this.img);

        const videoTexture = new THREE.VideoTexture(this.video);
        videoTexture.minFilter = THREE.LinearFilter;
        videoTexture.magFilter = THREE.LinearFilter;

        this.material = new THREE.MeshBasicMaterial({
            map: videoTexture,
            side: THREE.DoubleSide,
        });

        const geometry = new THREE.PlaneGeometry(sizeMeters, sizeMeters);
        this.mesh = new THREE.Mesh(geometry, this.material);
        this.mesh.rotation.x = -Math.PI / 2;
        this.mesh.position.y = yOffset;
        this.mesh.visible = false;
        scene.add(this.mesh);

        this.targetListManager = tlm;
        this.targetsRoot = new THREE.Group();
        this.targetsRoot.position.y = yOffset + 0.002; // clear of the projection quad
        scene.add(this.targetsRoot);
        this.ringGeometry = new THREE.RingGeometry(TARGET_RING_INNER, TARGET_RING_OUTER, 32);
        this.discGeometry = new THREE.CircleGeometry(TARGET_RING_OUTER, 32);

        this.provisionalRing = new THREE.Mesh(this.ringGeometry, new THREE.MeshBasicMaterial({
            color: TargetColors.mouse,
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide,
            depthWrite: false,
        }));
        this.provisionalRing.rotation.x = -Math.PI / 2;
        this.provisionalRing.visible = false;
        this.targetsRoot.add(this.provisionalRing);
    }

    // Preview where a target would be added, or null to take the preview down.
    // The point is a world-space floor position, as picked in the 3D view.
    public setProvisionalTarget(point: THREE.Vector3 | null) {
        if (!point) {
            this.provisionalRing.visible = false;
            return;
        }
        this.provisionalRing.position.set(point.x, 0, point.z);
        this.provisionalRing.visible = true;
    }

    // Show the current target list as rings on the floor.
    public updateTargets(targets: nf.telemetry.IOneTarget[]) {
        this.lastTargets = targets;
        this.renderTargets();
    }

    // Repaint with the current hover/selection state. Hover and selection are
    // global (owned by the TargetListManager), so a change made in any view —
    // the target list or either anchor cam — has to be reflected here too.
    public refresh() {
        this.renderTargets();
    }

    // Which target, if any, a point on the floor lands on. Done analytically
    // rather than by raycasting the ring meshes so that a click just inside or
    // beside the thin annulus still counts.
    public pickTarget(floorPoint: THREE.Vector3): string | null {
        let bestId: string | null = null;
        let bestDist = TARGET_PICK_RADIUS;
        for (const target of this.lastTargets) {
            if (!target.id) continue;
            const dx = (target.position?.x ?? 0) - floorPoint.x;
            const dz = -(target.position?.y ?? 0) - floorPoint.z;
            const dist = Math.hypot(dx, dz);
            if (dist < bestDist) {
                bestDist = dist;
                bestId = target.id;
            }
        }
        return bestId;
    }

    private renderTargets() {
        const hoveredId = this.targetListManager?.getHoveredId() ?? null;
        const selectedId = this.targetListManager?.getSelectedId() ?? null;

        this.lastTargets.forEach((target, i) => {
            if (i >= this.markerPool.length) this.growMarkerPool();
            const { ring, disc } = this.markerPool[i];

            // Ring colour tracks the target's status in the robot's queue.
            let color = TargetColors.seen;
            if (target.status == nf.telemetry.TargetStatus.TARGETSTATUS_SELECTED) {
                color = TargetColors.movingTo;
            } else if (target.status == nf.telemetry.TargetStatus.TARGETSTATUS_PICKED_UP) {
                color = TargetColors.grasped;
            }
            (ring.material as THREE.MeshBasicMaterial).color.set(color);

            // Positions are robot coordinates (Z-up), same conversion as elsewhere.
            const x = target.position?.x ?? 0;
            const z = -(target.position?.y ?? 0);
            ring.position.set(x, 0, z);
            disc.position.set(x, -0.0005, z); // just under the ring, avoids z-fighting
            ring.visible = true;

            // Fill matches the 2D overlays: orange when selected, blue on hover.
            const discMat = disc.material as THREE.MeshBasicMaterial;
            if (target.id && target.id === selectedId) {
                discMat.color.set(TargetColors.selected);
                discMat.opacity = 0.7;
                disc.visible = true;
            } else if (target.id && target.id === hoveredId) {
                discMat.color.set(TargetColors.hovered);
                discMat.opacity = 0.3;
                disc.visible = true;
            } else {
                disc.visible = false;
            }
        });

        for (let i = this.lastTargets.length; i < this.markerPool.length; i++) {
            this.markerPool[i].ring.visible = false;
            this.markerPool[i].disc.visible = false;
        }
    }

    private growMarkerPool() {
        const makeMesh = (geometry: THREE.BufferGeometry, opacity: number) => {
            const material = new THREE.MeshBasicMaterial({
                transparent: true,
                opacity,
                side: THREE.DoubleSide,
                depthWrite: false,
            });
            const mesh = new THREE.Mesh(geometry, material);
            mesh.rotation.x = -Math.PI / 2; // lay flat
            this.targetsRoot.add(mesh);
            return mesh;
        };

        this.markerPool.push({
            ring: makeMesh(this.ringGeometry, 0.9),
            disc: makeMesh(this.discGeometry, 0.3),
        });
    }

    private teardownHls() {
        if (this.hls) {
            this.hls.destroy();
            this.hls = null;
        }
    }

    public connectLocal(uri: string) {
        this.peerConnection?.close();
        this.peerConnection = null;
        this.teardownHls();
        this.video.srcObject = null;

        const imgTexture = new THREE.Texture(this.img);
        imgTexture.minFilter = THREE.LinearFilter;
        imgTexture.magFilter = THREE.LinearFilter;
        imgTexture.generateMipmaps = false;
        this.material.map = imgTexture;
        this.material.needsUpdate = true;

        const separator = uri.includes('?') ? '&' : '?';
        this.img.src = `${uri}${separator}t=${Date.now()}`;
        this.img.onload = () => {
            this.mesh.visible = true;
        };
        this.isLocalMode = true;
    }

    // Spectator path: play the floor-projection stream over HLS instead of WHEP.
    public async connectHLS(streamPath: string, ticket?: string) {
        this.isLocalMode = false;
        this.img.src = '';
        if (this.peerConnection) { this.peerConnection.close(); this.peerConnection = null; }
        this.teardownHls();

        let host = 'https://media.neufangled.com:8888';
        if (window.location.host.includes('localhost')) host = 'http://localhost:8888';
        const sep = (u: string) => u.includes('?') ? '&' : '?';
        let url = `${host}/${streamPath}/index.m3u8`;
        if (ticket) url += `${sep(url)}ticket=${encodeURIComponent(ticket)}`;
        if (window.location.host.includes('nf-site-monolith-staging')) url += `${sep(url)}staging=1`;

        const videoTexture = new THREE.VideoTexture(this.video);
        videoTexture.minFilter = THREE.LinearFilter;
        videoTexture.magFilter = THREE.LinearFilter;
        this.material.map = videoTexture;
        this.material.needsUpdate = true;

        const onReady = () => { this.video.play().catch(() => {}); this.mesh.visible = true; };

        if (this.video.canPlayType('application/vnd.apple.mpegurl')) {
            this.video.src = url;
            this.video.addEventListener('loadedmetadata', onReady, { once: true });
            return;
        }
        if (Hls.isSupported()) {
            const hls = new Hls({ lowLatencyMode: true });
            this.hls = hls;
            hls.loadSource(url);
            hls.attachMedia(this.video);
            hls.on(Hls.Events.MANIFEST_PARSED, onReady);
            hls.on(Hls.Events.ERROR, (_evt, data) => {
                if (data.fatal) console.warn(`FloorProjection HLS error for ${streamPath}:`, data.details);
            });
        } else {
            console.warn('HLS is not supported for floor projection in this browser.');
        }
    }

    public async connectWebRTC(streamPath: string, ticket?: string) {
        this.isLocalMode = false;
        this.img.src = '';
        this.teardownHls();

        let whepUrl = `https://media.neufangled.com:8889/${streamPath}/whep`;
        if (window.location.host.includes('localhost')) {
            whepUrl = `http://localhost:8889/${streamPath}/whep`;
        }
        const sep = (u: string) => u.includes('?') ? '&' : '?';
        if (ticket) whepUrl += `${sep(whepUrl)}ticket=${encodeURIComponent(ticket)}`;
        if (window.location.host.includes('nf-site-monolith-staging')) {
            whepUrl += `${sep(whepUrl)}staging=1`;
        }

        const videoTexture = new THREE.VideoTexture(this.video);
        videoTexture.minFilter = THREE.LinearFilter;
        videoTexture.magFilter = THREE.LinearFilter;
        this.material.map = videoTexture;
        this.material.needsUpdate = true;

        try {
            if (this.peerConnection) this.peerConnection.close();
            this.peerConnection = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
            });
            this.peerConnection.oniceconnectionstatechange = () => {
                console.log('FloorProjection ICE state:', this.peerConnection?.iceConnectionState);
            };
            this.peerConnection.ontrack = (event) => {
                if (event.track.kind === 'video') {
                    this.video.srcObject = event.streams[0];
                    this.video.play().catch(e => console.warn('FloorProjection play() failed:', e));
                    this.mesh.visible = true;
                }
            };
            this.peerConnection.addTransceiver('video', { direction: 'recvonly' });

            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);

            const response = await fetch(whepUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/sdp' },
                body: offer.sdp,
            });
            if (!response.ok) throw new Error(`WHEP ${response.status}`);
            await this.peerConnection.setRemoteDescription({
                type: 'answer',
                sdp: await response.text(),
            });
        } catch (err) {
            console.error('FloorProjection WebRTC error:', err);
        }
    }

    // Call once per animation frame — needed to push MJPEG img frames to the GPU.
    public update() {
        if (this.isLocalMode && this.mesh.visible) {
            (this.material.map as THREE.Texture).needsUpdate = true;
        }
    }

    public setOffline() {
        this.peerConnection?.close();
        this.peerConnection = null;
        this.video.srcObject = null;
        this.img.src = '';
        this.isLocalMode = false;
        this.mesh.visible = false;
        // Drop the markers too, so nothing stale stays on the floor or pickable.
        this.lastTargets = [];
        this.renderTargets();
        this.setProvisionalTarget(null);
    }
}
