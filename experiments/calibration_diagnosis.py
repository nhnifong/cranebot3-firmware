#!/usr/bin/env python
"""Explain what a recorded calibration run actually optimized, and what it cost.

Reads calibration_diagnostics.pkl (written by Observer._record_calibration_diagnostics when
--rec_diagnostics is on) and reports, for every optimizer pass:

  - the cost breakdown by term, with the residual count and per-residual rms behind each one,
    so one bad constraint reads differently from a thousand slightly-unhappy ones
  - what each pass changed: the same residual set scored at its input and its output
  - origin_consistency per marker, which locates a batch that spans a move
  - the gripper card term split into within-card pairs (vertical, so they compare the gripper
    camera against the encoders) and across-card pairs (which compare the anchor cameras'
    card layout against the encoders)
  - whether the diamond and gripper terms agree about room scale. They see the same encoders
    through the same cameras, so a disagreement is tag-specific rather than camera-wide
  - the resulting pull-point geometry, its coplanarity and its pairwise distances
  - why the run stopped and the step it was on, when it did not finish

  python experiments/calibration_diagnosis.py
  python experiments/calibration_diagnosis.py --pickle other_run.pkl --rerun
"""

import argparse
import contextlib
import io
import itertools
import logging
import pickle
import sys

import numpy as np
from scipy import optimize

import nf_robot.common.definitions as model_constants
from nf_robot.common.pose_functions import compose_poses
from nf_robot.host import eyelet_calibration as ec

N_ANCHORS = 2
LINE_NAMES = ['a0 direct', 'a0 external', 'a1 direct', 'a1 external']


def tilt_nodes(cam_tilts):
    """The per-camera extra-tilt nodes, exactly as multi_card_residuals builds them."""
    return [
        (np.array([(22 - t) / 180.0 * np.pi, 0, 0], dtype=float), np.zeros(3, dtype=float))
        for t in cam_tilts
    ]


def project_marker(name, sightings, anchor_poses, cam_tilts):
    """Every sighting of one marker, projected into the room at the given anchor poses."""
    nodes = tilt_nodes(cam_tilts)
    points = []
    for anchor_idx, poses in enumerate(sightings):
        for pose_cam in poses:
            if pose_cam is None or np.all(pose_cam == 0):
                continue
            chain = [anchor_poses[anchor_idx], model_constants.arp_anchor_camera,
                     nodes[anchor_idx], pose_cam]
            if name == 'gantry':
                chain.append(ec.gantry_april_inv)
            points.append((anchor_idx, compose_poses(chain)[1]))
    return points


def card_centroids(raw_obs, anchor_poses, cam_tilts):
    out = {}
    for name, sightings in raw_obs.items():
        points = project_marker(name, sightings, anchor_poses, cam_tilts)
        if points:
            out[name] = np.mean([p for _, p in points], axis=0)
    return out


def pull_points(anchor_poses, eyelet_positions):
    """The four points the lines leave the machine from, in firmware line order."""
    return np.array([
        compose_poses([anchor_poses[0], model_constants.arp_anchor_right_eyelet])[1],
        eyelet_positions[0],
        compose_poses([anchor_poses[1], model_constants.arp_anchor_right_eyelet])[1],
        eyelet_positions[1],
    ])


def split_state(x):
    return x[:12].reshape((2, 2, 3)), x[12:].reshape((2, 3))


def residual_counts(args, costs):
    """How many residuals sit behind each named cost. Mirrors the structure multi_card_residuals
    builds, and is cross-checked against the real residual vector so drift there is caught."""
    raw_obs = args['raw_obs']
    counts = {k: 0 for k in costs}

    for name, sightings in raw_obs.items():
        n = sum(1 for poses in sightings for p in poses if p is not None)
        if name == 'origin':
            counts['origin_consistency'] += 3 * n
        elif n > 1:
            counts['origin_consistency'] += 3 * n

    diamond = args.get('diamond_observations') or {}
    n_diamond_points = sum(
        1 for poses in diamond.values() for cam in poses for p in cam
        if p is not None and not np.all(p == 0)
    )
    counts['diamond_planarity'] = n_diamond_points
    counts['diamond_distances'] = 8 if all(
        s in diamond for s in ('bottom', 'right', 'top', 'left')) else 0

    gripper = args.get('gripper_obs') or {}
    n_samples = sum(len(v) for k, v in gripper.items() if k in raw_obs)
    counts['gripper_cards'] = 4 * n_samples * (n_samples - 1) // 2

    counts['shape_match'] = 1
    # always 6: optimize_arp_anchors builds its own eyelet guess when the caller passes none
    counts['eyelet_reg'] = 6
    counts['room_yaw'] = 1 if args.get('yaw_reference') is not None else 0
    if 'anchor_planarity' in counts:
        counts['anchor_planarity'] = 4
        counts['anchor_tilt'] = 6
    return counts


def evaluate(x, args):
    """multi_card_residuals at x with this pass's recorded inputs."""
    return ec.multi_card_residuals(
        x, args['raw_obs'], args.get('diamond_observations'),
        args.get('initial_eyelet_guesses'), False, args.get('fixed_anchor_poses'),
        args.get('line_deltas'), args.get('cam_tilts', (22, 22)), args.get('gripper_obs'),
        args.get('diamond_size', ec.DIAMOND_SIZE), args.get('yaw_reference'),
    )


def gripper_samples(args, anchor_poses, eyelet_positions):
    """(card name, gantry room position, line lengths) per hover, as the residual builds them."""
    centroids = card_centroids(args['raw_obs'], anchor_poses, args.get('cam_tilts', (22, 22)))
    out = []
    for name, samples in (args.get('gripper_obs') or {}).items():
        if name not in centroids:
            continue
        for obs in samples:
            out.append((name,
                        centroids[name] + np.asarray(obs['gantry_minus_card']),
                        np.asarray(obs['line_lengths'])))
    return out, centroids


def gripper_split(args, anchor_poses, eyelet_positions):
    """Within-card vs across-card fit of the gripper term, as (n, mean |modelled|,
    mean |measured|, slope, rms, cost) per group. slope is the least-squares gain of measured
    onto modelled; 1.0 means the two accounts of the same move agree in scale."""
    samples, _ = gripper_samples(args, anchor_poses, eyelet_positions)
    pulls = pull_points(anchor_poses, eyelet_positions)
    groups = {'within-card': ([], []), 'across-card': ([], [])}
    for i, j in itertools.combinations(range(len(samples)), 2):
        name_a, g_a, L_a = samples[i]
        name_b, g_b, L_b = samples[j]
        key = 'within-card' if name_a == name_b else 'across-card'
        for k in range(4):
            modelled = np.linalg.norm(g_b - pulls[k]) - np.linalg.norm(g_a - pulls[k])
            groups[key][0].append(modelled)
            groups[key][1].append(L_b[k] - L_a[k])

    stats = {}
    for key, (mod, meas) in groups.items():
        mod, meas = np.array(mod), np.array(meas)
        if len(mod) == 0:
            continue
        slope = float(np.dot(mod, meas) / np.dot(mod, mod)) if np.dot(mod, mod) > 0 else np.nan
        stats[key] = dict(n=len(mod), mean_model=np.abs(mod).mean(), mean_meas=np.abs(meas).mean(),
                          slope=slope, rms=float(np.sqrt(((mod - meas) ** 2).mean())),
                          cost=float(((mod - meas) ** 2).sum()))
    return stats


def best_card_layout_scale(args, anchor_poses, eyelet_positions):
    """The uniform horizontal scale on the card layout that best satisfies the encoders, with
    everything else held still. 1.0 means the cameras and the encoders agree."""
    samples, centroids = gripper_samples(args, anchor_poses, eyelet_positions)
    if len(samples) < 2:
        return None
    pulls = pull_points(anchor_poses, eyelet_positions)
    offsets = {name: np.asarray(obs['gantry_minus_card'])
               for name, obs_list in (args.get('gripper_obs') or {}).items() for obs in obs_list}

    def cost(scale):
        scaled = []
        for name, samples_for_card in (args.get('gripper_obs') or {}).items():
            if name not in centroids:
                continue
            pos = centroids[name].copy()
            pos[:2] *= scale
            for obs in samples_for_card:
                scaled.append((pos + np.asarray(obs['gantry_minus_card']),
                               np.asarray(obs['line_lengths'])))
        total = 0.0
        for i, j in itertools.combinations(range(len(scaled)), 2):
            for k in range(4):
                modelled = (np.linalg.norm(scaled[j][0] - pulls[k])
                            - np.linalg.norm(scaled[i][0] - pulls[k]))
                total += (modelled - (scaled[j][1][k] - scaled[i][1][k])) ** 2
        return total

    result = optimize.minimize_scalar(cost, bounds=(0.7, 1.3), method='bounded')
    return float(result.x), float(result.fun), float(cost(1.0))


def best_room_scale(args, anchor_poses, eyelet_positions):
    """The uniform scale on the whole reconstruction that best satisfies each length-delta
    term. They read the same encoders through the same cameras, so a disagreement is
    specific to one tag rather than a camera-wide scale error."""
    cam_tilts = args.get('cam_tilts', (22, 22))

    def scaled_state(s):
        A = np.array(anchor_poses, dtype=float)
        A[:, 1, :] *= s
        return A, np.array(eyelet_positions, dtype=float) * s

    def gripper_cost(s):
        A, E = scaled_state(s)
        pulls = pull_points(A, E)
        centroids = {n: c * s for n, c in
                     card_centroids(args['raw_obs'], anchor_poses, cam_tilts).items()}
        pts = [(centroids[n] + np.asarray(o['gantry_minus_card']), np.asarray(o['line_lengths']))
               for n, obs in (args.get('gripper_obs') or {}).items() if n in centroids for o in obs]
        total = 0.0
        for i, j in itertools.combinations(range(len(pts)), 2):
            for k in range(4):
                modelled = np.linalg.norm(pts[j][0] - pulls[k]) - np.linalg.norm(pts[i][0] - pulls[k])
                total += (modelled - (pts[j][1][k] - pts[i][1][k])) ** 2
        return total

    def diamond_cost(s):
        A, E = scaled_state(s)
        diamond = args.get('diamond_observations') or {}
        deltas = args.get('line_deltas')
        states = ('bottom', 'right', 'top', 'left')
        if not all(st in diamond for st in states) or deltas is None:
            return None
        centroids = {}
        for st in states:
            pts = [p for _, p in project_marker('gantry', diamond[st], anchor_poses, cam_tilts)]
            centroids[st] = np.mean(pts, axis=0) * s
        edges = [('bottom', 'right', 'bot_to_rig'), ('right', 'top', 'rig_to_top'),
                 ('top', 'left', 'top_to_lef'), ('left', 'bottom', None)]
        total = 0.0
        for eyelet_idx in (0, 1):
            measured = {k: deltas[k][eyelet_idx] for _, _, k in edges if k}
            measured[None] = -sum(measured.values())
            for a, b, key in edges:
                modelled = (np.linalg.norm(centroids[b] - E[eyelet_idx])
                            - np.linalg.norm(centroids[a] - E[eyelet_idx]))
                total += (modelled - measured[key]) ** 2
        return total

    out = {}
    for label, fn in (('gripper cards', gripper_cost), ('diamond', diamond_cost)):
        if fn(1.0) is None:
            continue
        r = optimize.minimize_scalar(fn, bounds=(0.7, 1.3), method='bounded')
        out[label] = (float(r.x), float(r.fun), float(fn(1.0)))
    return out


def rerun(record):
    """Reproduce a pass's output by re-running the optimizer on its recorded arguments."""
    args = dict(record['args'])
    args['time_budget_s'] = max(args.get('time_budget_s', 12.0), 120.0)
    # least_squares(verbose=1) prints straight to stdout, which would interleave with the report
    with contextlib.redirect_stdout(io.StringIO()):
        anchors, eyelets, floor_z, fit_info = ec.optimize_arp_anchors(**args)
    # back into the frame it solved in (origin card at z=0), to compare with the next warm start
    anchors = np.array(anchors, dtype=float)
    anchors[:, 1, 2] += floor_z
    eyelets = np.array(eyelets, dtype=float)
    eyelets[:, 2] += floor_z
    return anchors, eyelets, fit_info


def fmt_costs(costs, counts, total):
    lines = []
    for name, value in sorted(costs.items(), key=lambda kv: -float(kv[1])):
        value = float(value)
        n = counts.get(name, 0)
        share = 100.0 * value / total if total else 0.0
        rms = f'{np.sqrt(value / n) * 100:8.2f}' if n else '       -'
        lines.append(f'    {name:20s} {value:10.4f}  {share:5.1f}%  n={n:5d}  weighted rms {rms} cm')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pickle', default='calibration_diagnostics.pkl')
    parser.add_argument('--rerun', action='store_true',
                        help='re-run every pass to recover its output geometry, not just the '
                             'inputs the pickle records (slower, needs the same code version)')
    parser.add_argument('--quiet', action='store_true', help='silence optimizer logging')
    opts = parser.parse_args(argv)

    if opts.quiet or not opts.rerun:
        logging.disable(logging.CRITICAL)
    np.set_printoptions(precision=3, suppress=True)

    with open(opts.pickle, 'rb') as fh:
        records = pickle.load(fh)
    n_aborts = sum(1 for r in records if 'abort' in r)
    print(f'{opts.pickle}: {len(records) - n_aborts} pass(es)'
          f'{", aborted" if n_aborts else ""}\n')

    outputs = {}  # pass name -> (anchors, eyelets) in the solver frame

    for idx, record in enumerate(records):
        # the run recorded why it stopped and what it was doing, rather than another pass
        if 'abort' in record:
            print('=' * 78)
            print('DID NOT FINISH')
            print('=' * 78)
            # the reason is the line the operator was shown, so it carries its own wording
            print(f'  reason: {record["abort"]}')
            print(f'  stopped during: {record["step"]} ({record["percent_complete"]:.0f}% '
                  f'complete)')
            print('  everything above is what it got through before that.')
            if record.get('error'):
                print()
                print('  ' + record['error'].rstrip().replace('\n', '\n  '))
            print()
            continue

        args = dict(record['args'])
        name = record['pass']
        fit = record.get('fit_info')
        print('=' * 78)
        print(f'{name}')
        print('=' * 78)

        if fit is None:
            print('  no fit_info recorded (the pass did not finish)\n')
            continue

        costs = fit['costs']
        counts = residual_counts(args, costs)
        total = fit['total_cost']
        print(f'  converged={fit["converged"]} nfev={fit["nfev"]} total_cost={total:.4f}')
        print(f'  cost by term (share of total, and how badly one residual misses on average):')
        print(fmt_costs(costs, counts, total))

        warm_a = args.get('initial_anchor_guesses')
        warm_e = args.get('initial_eyelet_guesses')
        if warm_a is not None and warm_e is not None:
            x0 = np.concatenate([np.asarray(warm_a).flatten(), np.asarray(warm_e).flatten()])
            r0, c0 = evaluate(x0, args)
            n_check = sum(counts.values())
            if n_check != len(r0):
                print(f'  ! residual bookkeeping drifted: counted {n_check}, got {len(r0)}. '
                      f'rms figures above are unreliable; update residual_counts().')
            print(f'\n  what this pass changed (same residual set at its input vs its output):')
            print(f'    {"term":20s} {"input":>10s} {"output":>10s} {"change":>10s}')
            for term in sorted(costs, key=lambda t: -abs(float(costs[t]) - float(c0.get(t, 0.0)))):
                before, after = float(c0.get(term, 0.0)), float(costs[term])
                print(f'    {term:20s} {before:10.4f} {after:10.4f} {after - before:+10.4f}')
            print(f'    {"TOTAL":20s} {sum(map(float, c0.values())):10.4f} {total:10.4f} '
                  f'{total - sum(map(float, c0.values())):+10.4f}')
            # only worth listing when it is not just the previous pass's output re-stated
            if not any(np.allclose(warm_a, a) and np.allclose(warm_e, e)
                       for a, e in outputs.values()):
                outputs[f'input to {name}'] = (np.asarray(warm_a), np.asarray(warm_e))
        else:
            print('\n  cold start (no warm-start geometry recorded for this pass)')

        if opts.rerun:
            try:
                anchors, eyelets, _ = rerun(record)
                outputs[f'output of {name}'] = (anchors, eyelets)
            except Exception as exc:  # a stale pickle can fail against current code
                print(f'  ! rerun failed: {exc}')

        # ---- input data quality -------------------------------------------------
        geom = outputs.get(f'output of {name}') or outputs.get(f'input to {name}')
        if geom is None:
            print()
            continue
        anchors, eyelets = geom

        print('\n  origin_consistency broken down by marker. The residual treats each batch as')
        print('  repeated looks at one static point, so a batch that spans a move contributes a')
        print('  spread no geometry can remove, and its share of the cost is pure floor:')
        marker_costs = {}
        for marker, sightings in args['raw_obs'].items():
            points = project_marker(marker, sightings, anchors, args.get('cam_tilts', (22, 22)))
            if not points:
                continue
            P = np.array([p for _, p in points])
            if marker == 'origin':
                cost = float(np.sum((P * ec.W_ORIGIN) ** 2))
            else:
                cost = float(np.sum(((P - P.mean(axis=0)) * ec.W_CONSISTENCY) ** 2))
            marker_costs[marker] = (cost, P, points)
        floor_total = sum(c for c, _, _ in marker_costs.values())
        for marker, (cost, P, points) in sorted(marker_costs.items(), key=lambda kv: -kv[1][0]):
            spread = float(np.sqrt(((P - P.mean(axis=0)) ** 2).sum(axis=1).mean()))
            share = 100.0 * cost / floor_total if floor_total else 0.0
            per_cam = [np.mean([p for a, p in points if a == c], axis=0)
                       for c in range(N_ANCHORS) if any(a == c for a, _ in points)]
            disagree = (f'  cameras differ by {np.linalg.norm(per_cam[0] - per_cam[1]) * 100:5.1f} cm'
                        if len(per_cam) == N_ANCHORS else '')
            flag = '   <-- a moving batch, not a geometry error' if spread > 0.15 else ''
            print(f'    {marker:14s} n={len(P):3d}  cost {cost:8.3f} ({share:5.1f}%)  '
                  f'rms spread {spread * 100:6.1f} cm{disagree}{flag}')

        # ---- gripper term -------------------------------------------------------
        if args.get('gripper_obs'):
            print('\n  gripper card term, split by pair type:')
            print('    within-card pairs are a pure vertical move (the card position cancels), so')
            print('    they compare the gripper camera against the encoders. across-card pairs')
            print("    compare the anchor cameras' card layout against the encoders.")
            for key, s in gripper_split(args, anchors, eyelets).items():
                print(f'    {key:12s} n={s["n"]:4d}  |modelled| {s["mean_model"] * 100:6.1f} cm  '
                      f'|measured| {s["mean_meas"] * 100:6.1f} cm  slope {s["slope"]:.4f}  '
                      f'rms {s["rms"] * 100:5.2f} cm  cost {s["cost"]:.3f}')
            scale = best_card_layout_scale(args, anchors, eyelets)
            if scale:
                s, cost_at_s, cost_at_1 = scale
                print(f'    best uniform card-layout scale {s:.4f} '
                      f'(unweighted cost {cost_at_1:.3f} -> {cost_at_s:.3f}); '
                      f'1.0 means the cameras and the encoders agree about the card layout')

            room = best_room_scale(args, anchors, eyelets)
            if len(room) > 1:
                print('\n  best whole-room scale according to each length-delta term. These read the')
                print('  same encoders through the same cameras, so a split means the error is')
                print('  specific to one tag rather than a camera-wide scale error:')
                for label, (s, at_s, at_1) in room.items():
                    # a term that barely improves at its own best scale is not measuring scale
                    weak = '  (weak: cost barely moves, scale not determined by this term)' \
                        if at_1 and (at_1 - at_s) / at_1 < 0.1 else ''
                    print(f'    {label:14s} {s:.4f}  (cost {at_1:.4f} -> {at_s:.4f}){weak}')
        print()

    # ---- geometry evolution -----------------------------------------------------
    if outputs:
        print('=' * 78)
        print('PULL POINT GEOMETRY')
        print('=' * 78)
        for label, (anchors, eyelets) in outputs.items():
            P = pull_points(anchors, eyelets)
            span = P[:, 2].max() - P[:, 2].min()
            print(f'\n  {label}')
            for line_name, p in zip(LINE_NAMES, P):
                print(f'    {line_name:12s} {np.round(p, 3)}')
            print(f'    height spread {span * 100:5.1f} cm '
                  f'(all four should be near coplanar in a real room)')
            dists = {f'{i}-{j}': np.linalg.norm(P[i] - P[j]) for i, j in itertools.combinations(range(4), 2)}
            print('    pairwise: ' + '  '.join(f'{k} {v:.3f}' for k, v in dists.items()))

        labels = list(outputs)
        for prev, cur in zip(labels, labels[1:]):
            P0, P1 = pull_points(*outputs[prev]), pull_points(*outputs[cur])
            moves = [np.linalg.norm(b - a) * 100 for a, b in zip(P0, P1)]
            print(f'\n  {prev} -> {cur}: pull points moved ' +
                  '  '.join(f'{n} {m:.1f}cm' for n, m in zip(LINE_NAMES, moves)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
