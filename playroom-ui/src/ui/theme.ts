import * as THREE from 'three';

// The palette lives in playroom-styles.css, one block of custom properties per
// scheme, selected by data-theme on <html>. CSS applies it to the DOM by itself;
// three.js can't read custom properties, so anything drawn in the 3D view takes
// its colors from themeColor() and re-reads them on every scheme change.

/** The schemes defined in playroom-styles.css, in the order the theme menu lists
 *  them. A new block over there is a new entry here and nowhere else. */
export const SCHEMES = [
    { id: 'light', label: 'Light' },
    { id: 'beige', label: 'Beige' },
    { id: 'dark',  label: 'Dark' },
    { id: 'loud',  label: 'Loud' },
];

const STORAGE_KEY = 'playroom-theme';

const prefersDark = window.matchMedia('(prefers-color-scheme: dark)');
const listeners: Array<() => void> = [];

// Set by setTheme; while it is null the browser's preference decides.
let pinned: string | null = null;

function apply() {
    document.documentElement.dataset.theme = pinned ?? (prefersDark.matches ? 'dark' : 'light');
    listeners.forEach(fn => fn());
}

/** Restores the stored choice, or follows the browser's dark mode preference. */
export function initTheme() {
    const stored = readStored();
    pinned = SCHEMES.some(s => s.id === stored) ? stored : null;
    apply();
    prefersDark.addEventListener('change', apply);
    if (!rawColor('--scene-bg')) {
        // Nothing else says so: the 3D view just comes up black.
        console.warn('theme: playroom-styles.css is not loaded, 3D colors are falling back to black');
    }
    // Reachable from the console for trying schemes out against a live scene.
    (window as any).setTheme = setTheme;
}

/** Pins a scheme by id, or null to follow the browser. Remembered across reloads. */
export function setTheme(id: string | null) {
    pinned = id;
    try {
        if (id) localStorage.setItem(STORAGE_KEY, id);
        else localStorage.removeItem(STORAGE_KEY);
    } catch { /* private browsing; the choice just won't outlive the tab */ }
    apply();
}

/** The pinned scheme's id, or null while following the browser. */
export function pinnedTheme(): string | null {
    return pinned;
}

/** The scheme actually in effect, pinned or not. */
export function activeTheme(): string {
    return document.documentElement.dataset.theme ?? 'light';
}

/** Runs fn on every scheme change, for colors held outside the DOM. */
export function onThemeChange(fn: () => void) {
    listeners.push(fn);
}

function readStored(): string | null {
    try {
        return localStorage.getItem(STORAGE_KEY);
    } catch {
        return null;
    }
}

function rawColor(name: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** One palette entry as a three.js color. */
export function themeColor(name: string): THREE.Color {
    return new THREE.Color(rawColor(name) || '#000000');
}
