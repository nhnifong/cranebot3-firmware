/**
 * Gating UI features and changes on the nf_robot version of the host this UI is connected to.
 *
 * One UI build is served to robots running hosts of different ages. Anything a newer host makes
 * redundant, or that only a newer host can accept, is keyed here by the first host version that
 * has it, and checked with hostSupports() at the point of use.
 */

// feature key -> minimum host nf_robot version that has it
export const HOST_VERSION_GATES = {
  // calibration measures camera tilt off the calibration cards and the pole off its swing period
  calibrationMeasuresTiltAndPole: '6.4.0',
} as const;

export type HostFeature = keyof typeof HOST_VERSION_GATES;

let hostVersion: string | null = null;

/** Record the version the host reported. Called only with a version actually received: setup
 * telemetry carries it, and other messages of the same type leave it out. */
export function setHostVersion(version: string) {
  hostVersion = version;
}

export function getHostVersion(): string | null {
  return hostVersion;
}

// A PEP 440 pre-release or dev segment directly after the release numbers. Ranked below the
// release it precedes, the same as the host's own packaging.version comparison, so a 6.4.0.dev1
// host does not pass a gate its code may not yet meet.
const PRE_RELEASE = /^[.\-_]?(a|b|c|rc|alpha|beta|pre|preview|dev)\d*/i;

interface ParsedVersion {
  release: [number, number, number];
  preRelease: boolean;
}

/** The release numbers of a version string and whether it is a pre-release, or null if it has
 * no numeric release at all. Post-release and local suffixes rank as the release itself. */
export function parseVersion(version: string): ParsedVersion | null {
  const m = /^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?/i.exec(version);
  if (!m) return null;
  return {
    release: [Number(m[1]), Number(m[2] ?? 0), Number(m[3] ?? 0)],
    preRelease: PRE_RELEASE.test(version.slice(m[0].length)),
  };
}

export function versionAtLeast(version: string, minimum: string): boolean {
  const have = parseVersion(version);
  const need = parseVersion(minimum);
  if (!have || !need) return false;
  for (let i = 0; i < 3; i++) {
    if (have.release[i] !== need.release[i]) return have.release[i] > need.release[i];
  }
  return !have.preRelease || need.preRelease;
}

/** Whether the connected host has a feature. A host that has not reported a version counts as
 * not having it: the hosts that never send one are precisely the ones older than the field. */
export function hostSupports(feature: HostFeature): boolean {
  return hostVersion != null && versionAtLeast(hostVersion, HOST_VERSION_GATES[feature]);
}
