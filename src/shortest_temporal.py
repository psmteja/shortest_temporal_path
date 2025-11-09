import os
import math
import json
import argparse
import itertools
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
from dateutil import parser as dtparse
from sgp4.api import Satrec, jday

# ------------------------------------------------------------
# Space-Track helpers
# ------------------------------------------------------------

SPACE_TRACK_BASE = "https://www.space-track.org"

def spacetrack_session(user: str, password: str) -> requests.Session:
    s = requests.Session()
    login_url = f"{SPACE_TRACK_BASE}/ajaxauth/login"
    data = {"identity": user, "password": password}
    r = s.post(login_url, data=data, timeout=30)
    r.raise_for_status()
    if "Session" not in r.text and r.status_code != 200:
        raise RuntimeError("Space-Track login failed")
    return s

def fetch_tles(
    sess: requests.Session,
    *,
    object_ids: Optional[List[int]] = None,
    object_like: Optional[str] = None,
    epoch_start: datetime,
    epoch_end: datetime,
) -> List[dict]:
    """
    Returns a list of TLE rows (Space-Track REST JSON).
    """
    base = f"{SPACE_TRACK_BASE}/basicspacedata/query/class/tle"
    epoch_str = f"/EPOCH/{epoch_start.strftime('%Y-%m-%d %H:%M:%S')}--{epoch_end.strftime('%Y-%m-%d %H:%M:%S')}"
    if object_ids:
        ids_clause = ",".join(str(x) for x in object_ids)
        q = f"/NORAD_CAT_ID/{ids_clause}"
    elif object_like:
        # OBJECT_NAME like 'STARLINK%' etc.
        q = f"/OBJECT_NAME/{object_like}"
    else:
        raise ValueError("Provide object_ids or object_like")

    url = f"{base}{q}{epoch_str}/orderby/NORAD_CAT_ID asc/format/json"
    r = sess.get(url, timeout=120)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Space-Track response: {data}")
    return data

def group_latest_tles_by_sat(tle_rows: List[dict]) -> Dict[int, Tuple[str, str]]:
    """
    Choose the most recent TLE per NORAD_CAT_ID within the fetched range.
    Returns {norad_id: (line1, line2)}.
    """
    latest: Dict[int, Tuple[datetime, str, str]] = {}
    for row in tle_rows:
        try:
            norad = int(row["NORAD_CAT_ID"])
            epoch = dtparse.parse(row["EPOCH"])
            l1 = row["TLE_LINE1"].strip()
            l2 = row["TLE_LINE2"].strip()
        except Exception:
            continue
        prev = latest.get(norad)
        if prev is None or epoch > prev[0]:
            latest[norad] = (epoch, l1, l2)
    return {k: (v[1], v[2]) for k, v in latest.items()}

# ------------------------------------------------------------
# Orbit propagation & geometry (TEME/ECI distance for simplicity)
# ------------------------------------------------------------

def sgp4_satrec_from_tle(line1: str, line2: str) -> Satrec:
    return Satrec.twoline2rv(line1, line2)

def eci_position_km(sat: Satrec, t: datetime) -> np.ndarray:
    """Return ECI position (TEME) vector [km] at UTC datetime t."""
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        # When sgp4 fails, return NaNs; caller can handle
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return np.array(r, dtype=float)  # km

def pairwise_distance_km(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))

# ------------------------------------------------------------
# Temporal-graph data structures (same semantics as your thesis)
# ------------------------------------------------------------

@dataclass(frozen=True)
class EdgeInfo:
    connectivity: List[int]  # 0/1 over M time slots
    t: int                   # consecutive 1s required to traverse
    d: Optional[float] = None

@dataclass
class PathEntry:
    start: int
    finish: int
    pred: int

def _find_all_runs_of_ones(cs: List[int]) -> List[Tuple[int, int]]:
    runs, i, n = [], 0, len(cs)
    while i < n:
        if cs[i] == 1:
            j = i
            while j + 1 < n and cs[j + 1] == 1:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs

def _all_valid_starts_for_length(cs: List[int], min_len: int, after_idx: int = -1) -> List[int]:
    if min_len <= 0:
        return []
    starts: List[int] = []
    for L, R in _find_all_runs_of_ones(cs):
        if (R - L + 1) < min_len:
            continue
        j0 = max(L, after_idx + 1)
        j1 = R - min_len + 1
        if j0 <= j1:
            starts.extend(range(j0, j1 + 1))
    return starts

def _insert_unique_entry(entries: List[PathEntry], new_entry: PathEntry) -> bool:
    for e in entries:
        if e.start == new_entry.start and e.finish == new_entry.finish:
            return False
    # prefer shorter durations for same finish
    for i, e in enumerate(entries):
        if e.finish == new_entry.finish and e.start < new_entry.start:
            entries[i] = new_entry
            return True
    entries.append(new_entry)
    entries.sort(key=lambda x: (x.finish, x.start))
    return True

def initialize_path_matrix(A: List[List[Optional[EdgeInfo]]]) -> List[List[List[PathEntry]]]:
    n = len(A)
    P: List[List[List[PathEntry]]] = [[[] for _ in range(n)] for __ in range(n)]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            info = A[p][q]
            if info is None:
                continue
            for j in _all_valid_starts_for_length(info.connectivity, info.t, -1):
                _insert_unique_entry(P[p][q], PathEntry(start=j, finish=j + info.t - 1, pred=p))
    return P

def _relax_through_q(A, P, p: int, q: int, r: int) -> bool:
    info = A[q][r]
    if info is None:
        return False
    changed = False
    t_qr, cs_qr = info.t, info.connectivity
    for epq in P[p][q]:
        for j in _all_valid_starts_for_length(cs_qr, t_qr, after_idx=epq.finish):
            changed |= _insert_unique_entry(P[p][r], PathEntry(epq.start, j + t_qr - 1, pred=q))
    return changed

def forward_pass(A, P) -> bool:
    n = len(A)
    any_change = False
    for q in range(n):
        for p in range(n):
            if p == q:
                continue
            for r in range(n):
                if r == q or r == p:
                    continue
                any_change |= _relax_through_q(A, P, p, q, r)
    return any_change

def backward_pass(A, P) -> bool:
    n = len(A)
    any_change = False
    for q in range(n - 1, -1, -1):
        for p in range(n):
            if p == q:
                continue
            for r in range(n):
                if r == q or r == p:
                    continue
                any_change |= _relax_through_q(A, P, p, q, r)
    return any_change

def reduce_to_shortest_time(P: List[List[List[PathEntry]]]) -> List[List[Optional[PathEntry]]]:
    n = len(P)
    R: List[List[Optional[PathEntry]]] = [[None for _ in range(n)] for __ in range(n)]
    for p in range(n):
        for r in range(n):
            cand = P[p][r]
            if not cand:
                continue
            R[p][r] = min(cand, key=lambda e: (e.finish - e.start + 1, e.finish, -e.start))
    return R

def reconstruct_path_nodes(p: int, r: int, best: PathEntry,
                           P: List[List[List[PathEntry]]],
                           A: List[List[Optional[EdgeInfo]]]) -> List[int]:
    path = [r]
    curr_target, curr_entry = r, best
    while curr_target != p:
        q = curr_entry.pred
        path.append(q)
        t_qr = A[q][curr_target].t
        j = curr_entry.finish + 1 - t_qr
        candidates = [e for e in P[p][q] if e.start == curr_entry.start and e.finish < j]
        if not candidates:
            break
        prev = max(candidates, key=lambda e: e.finish)
        curr_target, curr_entry = q, prev
    path.append(p)
    path.reverse()
    return path

def all_pairs_shortest_time_paths(A: List[List[Optional[EdgeInfo]]]):
    P = initialize_path_matrix(A)
    forward_pass(A, P)
    backward_pass(A, P)
    best = reduce_to_shortest_time(P)
    n = len(A)
    best_paths: Dict[Tuple[int, int], List[int]] = {}
    for p in range(n):
        for r in range(n):
            if best[p][r] is None:
                continue
            best_paths[(p, r)] = reconstruct_path_nodes(p, r, best[p][r], P, A)
    return best, best_paths

# ------------------------------------------------------------
# Build temporal graph from propagated satellites
# ------------------------------------------------------------

def build_connectivity_strings(
    norad_ids: List[int],
    sat_pos_times: Dict[int, List[np.ndarray]],
    dist_thresh_km: float,
) -> Tuple[List[List[Optional[EdgeInfo]]], Dict[int, int]]:
    """
    Create A[p][q] EdgeInfo with binary connectivity strings:
      edge (p->q) is '1' at slot m iff distance(p,q,m) <= dist_thresh_km.
    Uses the same 't' for all edges (set later by caller).
    We index satellites in the order they appear in norad_ids.
    """
    n = len(norad_ids)
    M = len(next(iter(sat_pos_times.values())))
    idx_of = {norad: i for i, norad in enumerate(norad_ids)}
    A: List[List[Optional[EdgeInfo]]] = [[None for _ in range(n)] for __ in range(n)]

    # Precompute distances per slot to avoid repeated work
    # positions[i][m] = ECI position for sat i at slot m
    positions = [sat_pos_times[nid] for nid in norad_ids]

    for i, j in itertools.permutations(range(n), 2):
        cs = [0] * M
        for m in range(M):
            pi = positions[i][m]
            pj = positions[j][m]
            if np.any(np.isnan(pi)) or np.any(np.isnan(pj)):
                cs[m] = 0
                continue
            d = pairwise_distance_km(pi, pj)
            cs[m] = 1 if d <= dist_thresh_km else 0
        # t (consecutive ones needed) will be set by caller uniformly
        A[i][j] = EdgeInfo(connectivity=cs, t=1)
    return A, idx_of

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Space-Track -> Temporal shortest path")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--object-ids", nargs="+", type=int, help="List of NORAD_CAT_IDs")
    g.add_argument("--object-like", type=str, help="OBJECT_NAME like pattern e.g. STARLINK%%")
    ap.add_argument("--start", required=True, help="UTC start ISO8601")
    ap.add_argument("--end", required=True, help="UTC end ISO8601")
    ap.add_argument("--step-min", type=int, default=2, help="Time-step in minutes (slot length τ)")
    ap.add_argument("--t-slots", type=int, default=2, help="Consecutive 1s required to traverse an edge")
    ap.add_argument("--dist-km", type=float, default=900.0, help="Max distance for a link to be 'up'")
    ap.add_argument("--src", type=int, required=True, help="Source NORAD_CAT_ID for path")
    ap.add_argument("--dst", type=int, required=True, help="Destination NORAD_CAT_ID for path")
    ap.add_argument("--max-sats", type=int, default=60, help="Optional cap for number of sats to keep")

    args = ap.parse_args()

    # Credentials
    user = os.getenv("SPACETRACK_USER")
    pw = os.getenv("SPACETRACK_PASS")
    if not user or not pw:
        raise SystemExit("Set SPACETRACK_USER and SPACETRACK_PASS environment variables.")

    t0 = dtparse.parse(args.start).astimezone(timezone.utc)
    t1 = dtparse.parse(args.end).astimezone(timezone.utc)
    if t1 <= t0:
        raise SystemExit("--end must be after --start")
    step = timedelta(minutes=args.step_min)

    # Login & fetch TLEs
    sess = spacetrack_session(user, pw)
    tle_rows = fetch_tles(
        sess,
        object_ids=args.object_ids,
        object_like=args.object_like,
        epoch_start=t0 - timedelta(days=3),  # small buffer
        epoch_end=t1 + timedelta(days=3),
    )
    latest = group_latest_tles_by_sat(tle_rows)
    norad_ids = sorted(latest.keys())
    if args.max_sats and len(norad_ids) > args.max_sats:
        norad_ids = norad_ids[: args.max_sats]

    if args.src not in norad_ids or args.dst not in norad_ids:
        raise SystemExit(f"src/dst NORAD IDs must be among fetched sats. Got {len(norad_ids)} sats.")

    # Build SGP4 propagators
    satrecs: Dict[int, Satrec] = {}
    for nid in norad_ids:
        l1, l2 = latest[nid]
        satrecs[nid] = sgp4_satrec_from_tle(l1, l2)

    # Time grid
    times: List[datetime] = []
    t = t0
    while t <= t1:
        times.append(t.replace(tzinfo=timezone.utc))
        t += step
    M = len(times)
    print(f"[info] Propagating {len(norad_ids)} satellites over {M} slots (τ={args.step_min} min)")

    # Propagate
    sat_pos_times: Dict[int, List[np.ndarray]] = {nid: [] for nid in norad_ids}
    for m, tm in enumerate(times):
        for nid in norad_ids:
            sat_pos_times[nid].append(eci_position_km(satrecs[nid], tm))

    # Build connectivity strings and temporal graph
    A, idx_of = build_connectivity_strings(norad_ids, sat_pos_times, args.dist_km)

    # Set uniform t (consecutive 1s needed) across edges
    for i, j in itertools.permutations(range(len(norad_ids)), 2):
        if A[i][j] is not None:
            A[i][j] = EdgeInfo(connectivity=A[i][j].connectivity, t=args.t_slots)

    # Compute all-pairs shortest temporal paths
    best, best_paths = all_pairs_shortest_time_paths(A)

    si, di = idx_of[args.src], idx_of[args.dst]
    be = best[si][di]
    if be is None:
        print(f"No temporal path from {args.src} to {args.dst} in the window.")
        return

    seq_idx = best_paths[(si, di)]
    seq_norad = [norad_ids[k] for k in seq_idx]
    duration_slots = be.finish - be.start + 1
    start_time = times[be.start]
    finish_time = times[be.finish]
    print("\n=== Shortest Temporal Path ===")
    print(f"From {args.src} to {args.dst}")
    print(f"Node sequence (NORAD_CAT_IDs): {seq_norad}")
    print(f"Start slot: {be.start} @ {start_time.isoformat()}Z")
    print(f"Finish slot: {be.finish} @ {finish_time.isoformat()}Z")
    print(f"Elapsed time: {duration_slots} slots  (~{duration_slots * args.step_min} minutes)")
    print(f"Slot length τ: {args.step_min} minutes, t (consecutive 1s required) = {args.t_slots}")
    print(f"Distance threshold: ≤ {args.dist_km} km defines a link-up")

if __name__ == "__main__":
    main()
