import * as THREE from 'three';
import Hls from 'hls.js';

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

    constructor(scene: THREE.Scene, sizeMeters: number = 5, yOffset: number = 0.001) {
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
    }
}
