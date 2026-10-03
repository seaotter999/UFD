#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build all_scanvp_cands.json for R2R/REVERIE-like tasks.

Output schema:
{
  "<scanId>_<viewpointId>": {
    "<neighbor_viewpointId>": [viewidx, rel_angle_dist, rel_heading, rel_elevation],
    ...
  },
  ...
}

- viewidx: best panoramic slice index in {0..35}
    0..11 : elevation -30°, heading 0..330° step 30°
    12..23: elevation   0°, heading 0..330° step 30°
    24..35: elevation +30°, heading 0..330° step 30°
- rel_heading/rel_elevation: relative angles (radians) in the camera frame
  under the chosen viewidx; small values like ~0.0x–0.2
- rel_angle_dist = sqrt(rel_heading^2 + rel_elevation^2)

Notes:
- Requires MatterSim (same as your project).
- We purposely do NOT use loc.x/y/z; the community files store *angles*, not positions.
"""

import os
import json
import math
import argparse

import MatterSim


def load_scan_viewpoints(connectivity_dir):
    """
    Read all scan ids and their included viewpoints from *_connectivity.json.
    Return: dict {scanId: [viewpointId, ...]}
    """
    scans = []
    for fname in os.listdir(connectivity_dir):
        if fname.endswith("_connectivity.json"):
            scans.append(fname.replace("_connectivity.json", ""))

    scan2vps = {}
    for scan in scans:
        path = os.path.join(connectivity_dir, f"{scan}_connectivity.json")
        with open(path, "r") as f:
            data = json.load(f)
        # keep included viewpoints only
        vps = [d["image_id"] for d in data if d.get("included", True)]
        scan2vps[scan] = vps
    return scan2vps


def best_candidates_for_viewpoint(sim, scanId, viewpointId):
    """
    Produce mapping:
      {neighbor_vpid: [best_viewidx, rel_angle_dist, rel_heading, rel_elevation]}

    This mirrors the logic used in common R2R implementations:
    pick the viewidx whose camera-relative angles to the neighbor are closest to (0,0).
    All angles are radians.
    """
    def _dist(h, e):
        return (h * h + e * e) ** 0.5

    best = {}  # vpid -> (dist, ix, rel_h, rel_e)

    for ix in range(36):
        if ix == 0:
            sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])      # switch elevation layer (-30 -> 0 -> +30)
        else:
            sim.makeAction([0], [1.0], [0])        # rotate heading +30°

        state = sim.getState()[0]
        assert state.viewIndex == ix

        # base angles of this slice (consistent with common R2R code)
        base_heading   = (ix % 12) * math.radians(30)
        base_elevation = (ix // 12 - 1) * math.radians(30)

        # align world angles into the local camera frame of this view slice
        heading   = state.heading   - base_heading
        elevation = state.elevation - base_elevation

        # skip index 0 (self), keep neighbors only
        for loc in state.navigableLocations[1:]:
            rel_h = heading   + loc.rel_heading
            rel_e = elevation + loc.rel_elevation
            d = _dist(rel_h, rel_e)

            vpid = loc.viewpointId
            cur = best.get(vpid)
            if (cur is None) or (d < cur[0]):
                best[vpid] = (d, ix, rel_h, rel_e)

    out = {}
    for vpid, (d, ix, rel_h, rel_e) in best.items():
        out[vpid] = [int(ix), float(d), float(rel_h), float(rel_e)]
    return out


def build_all_scanvp_cands(connectivity_dir, output_path, scans_subset=None,
                           image_w=640, image_h=480, vfov=60):
    """
    Main routine:
    - init single MatterSim env (discretized angles)
    - iterate all viewpoints to build mapping
    - write JSON
    """
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)  # 30° bins
    sim.setCameraResolution(image_w, image_h)
    sim.setCameraVFOV(math.radians(vfov))
    sim.setBatchSize(1)
    sim.initialize()

    scan2vps = load_scan_viewpoints(connectivity_dir)
    if scans_subset:
        keep = set(scans_subset)
        scan2vps = {k: v for k, v in scan2vps.items() if k in keep}

    total = sum(len(vps) for vps in scan2vps.values())
    done = 0
    result = {}

    for scan, vps in scan2vps.items():
        for vp in vps:
            key = f"{scan}_{vp}"
            result[key] = best_candidates_for_viewpoint(sim, scan, vp)

            done += 1
            if done % 100 == 0 or done == total:
                print(f"[{done}/{total}] processed")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f)

    print(f"Saved {len(result)} entries to {output_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Build all_scanvp_cands.json (viewidx + relative angles per edge)."
    )
    p.add_argument("--connectivity_dir", type=str, required=True,
                   help="Directory containing *_connectivity.json files.")
    p.add_argument("--output", type=str, required=True,
                   help="Output path for all_scanvp_cands.json")
    p.add_argument("--scans", type=str, default="",
                   help="Optional comma-separated scan ids to restrict "
                        "(e.g. 17DRP5sb8fy,gZ6f7yhEvPG)")
    return p.parse_args()


def main():
    args = parse_args()
    scans_subset = [s.strip() for s in args.scans.split(",")] if args.scans else None
    build_all_scanvp_cands(
        connectivity_dir=args.connectivity_dir,
        output_path=args.output,
        scans_subset=scans_subset
    )


if __name__ == "__main__":
    main()