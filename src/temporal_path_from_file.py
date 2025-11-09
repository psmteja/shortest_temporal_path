#!/usr/bin/env python3
"""
Hardcoded run of the temporal-shortest-path pipeline using a downloaded
Space-Track JSON file (e.g., gp/tle_last30). No argparse—edit CONFIG below.

Tip:
- Set PRINT_SATS = True for a quick list of available NORAD IDs and names.
"""

# ========================= CONFIG (edit me) =========================
JSON_PATH   = "tle_recent_30.json"      # Path to your downloaded JSON
START_ISO   = "2025-11-08T00:00:00Z"    # UTC start time (ISO8601)
END_ISO     = "2025-11-08T02:00:00Z"    # UTC end time   (ISO8601)

STEP_MIN    = 2       # slot length τ in minutes
T_SLOTS     = 2       # consecutive 'up' slots required to traverse an edge
DIST_KM     = 900.0   # link is 'up' when distance <= DIST_KM

# Pick two IDs that exist in your JSON file. Examples from your sample dump:
SRC_ID      = 5       # e.g., VANGUARD 1
DST_ID      = 11      # e.g., VANGUARD 2

OBJECT_LIKE = None    # e.g., "STARLINK" to filter by name (substring, case-insensitive)
MAX_SATS    = 0       # 0 = no cap; otherwise keep only first N satellites (sorted by NORAD)
PRINT_SATS  = False   # True = just list NORADs & names then exit
# ===================================================================

import json
import itertools
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from dateutil import parser as dtparse
from sgp4.api import Satrec, jday

# -------------------------- Temporal graph structs ---------------------------

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

# --------------------------- SGP4 helpers -----------------------------------

def sgp4_satrec_from_tle(line1: str, line2: str) -> Satrec:
    return Satrec.twoline2rv(line1, line2)

def eci_position_km(sat: Satrec, t: datetime) -> np.ndarray:
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return np.array(r, dtype=float)

def pairwise_distance_km(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))

# --------------------------- JSON loader ------------------------------------

def load_latest_tles_by_sat(json_path: str, object_like: Optional[str]) -> Dict[int, Tuple[str, str, str]]:
    """
    Load a Space-Track JSON array and keep the most recent record per NORAD_CAT_ID.
    Returns {norad_id: (OBJECT_NAME, TLE_LINE1, TLE_LINE2)}.
    If object_like is provided, filter OBJECT_NAME contains case-insensitive substring.
    Supports rows that include TLE_LINE1/2 (gp, gp_history, tle, tle_latest).
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array.")

    filt = (object_like or "").lower()
    latest: Dict[int, Tuple[datetime, str, str, str]] = {}
    for row in data:
        try:
            name = str(row.get("OBJECT_NAME") or row.get("SATNAME") or "").strip()
            if filt and filt not in name.lower():
                continue
            norad = int(row["NORAD_CAT_ID"] if "NORAD_CAT_ID" in row else row["OBJECT_NUMBER"])
            epoch = dtparse.parse(row.get("EPOCH") or row.get("EPOCH_E", ""))
            l1 = row["TLE_LINE1"].strip()
            l2 = row["TLE_LINE2"].strip()
        except Exception:
            continue
        prev = latest.get(norad)
        if prev is None or epoch > prev[0]:
            latest[norad] = (epoch, name, l1, l2)

    return {k: (v[1], v[2], v[3]) for k, v in latest.items()}

# --------------------------- Build temporal graph ---------------------------

def build_connectivity_strings(
    norad_ids: List[int],
    sat_pos_times: Dict[int, List[np.ndarray]],
    dist_thresh_km: float,
) -> Tuple[List[List[Optional[EdgeInfo]]], Dict[int, int]]:
    n = len(norad_ids)
    M = len(next(iter(sat_pos_times.values())))
    idx_of = {norad: i for i, norad in enumerate(norad_ids)}
    A: List[List[Optional[EdgeInfo]]] = [[None for _ in range(n)] for __ in range(n)]

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
        A[i][j] = EdgeInfo(connectivity=cs, t=1)
    return A, idx_of

# --------------------------- Driver -----------------------------------------

def main():
    t0 = dtparse.parse(START_ISO).astimezone(timezone.utc)
    t1 = dtparse.parse(END_ISO).astimezone(timezone.utc)
    if t1 <= t0:
        raise SystemExit("END_ISO must be after START_ISO")
    step = timedelta(minutes=STEP_MIN)

    latest = load_latest_tles_by_sat(JSON_PATH, OBJECT_LIKE)
    if not latest:
        raise SystemExit("No usable TLEs found in the JSON (after filtering).")

    norad_ids = sorted(latest.keys())
    if MAX_SATS and len(norad_ids) > MAX_SATS:
        norad_ids = norad_ids[:MAX_SATS]

    if PRINT_SATS:
        print("Available satellites (NORAD_CAT_ID: OBJECT_NAME):")
        for nid in norad_ids:
            name, _, _ = latest[nid]
            print(f"{nid}: {name}")
        return

    if SRC_ID not in norad_ids or DST_ID not in norad_ids:
        raise SystemExit("SRC_ID/DST_ID must be among the filtered/capped set. Set PRINT_SATS=True to inspect.")

    # Build SGP4 propagators
    satrecs: Dict[int, Satrec] = {}
    for nid in norad_ids:
        name, l1, l2 = latest[nid]
        satrecs[nid] = sgp4_satrec_from_tle(l1, l2)

    # Time grid
    times: List[datetime] = []
    t = t0
    while t <= t1:
        times.append(t.replace(tzinfo=timezone.utc))
        t += step
    M = len(times)
    print(f"[info] Propagating {len(norad_ids)} satellites over {M} slots (τ={STEP_MIN} min)")

    # Propagate
    sat_pos_times: Dict[int, List[np.ndarray]] = {nid: [] for nid in norad_ids}
    for tm in times:
        for nid in norad_ids:
            sat_pos_times[nid].append(eci_position_km(satrecs[nid], tm))

    # Temporal graph
    A, idx_of = build_connectivity_strings(norad_ids, sat_pos_times, DIST_KM)
    for i, j in itertools.permutations(range(len(norad_ids)), 2):
        if A[i][j] is not None:
            A[i][j] = EdgeInfo(connectivity=A[i][j].connectivity, t=T_SLOTS)

    # Shortest temporal path
    best, best_paths = all_pairs_shortest_time_paths(A)

    si, di = idx_of[SRC_ID], idx_of[DST_ID]
    be = best[si][di]
    if be is None:
        print(f"No temporal path from {SRC_ID} to {DST_ID} in the window.")
        return

    seq_idx = best_paths[(si, di)]
    seq_norad = [norad_ids[k] for k in seq_idx]
    duration_slots = be.finish - be.start + 1
    start_time = times[be.start]
    finish_time = times[be.finish]

    print("\n=== Shortest Temporal Path ===")
    print(f"From {SRC_ID} to {DST_ID}")
    print(f"Node sequence (NORAD_CAT_IDs): {seq_norad}")
    print(f"Start slot: {be.start} @ {start_time.isoformat()}Z")
    print(f"Finish slot: {be.finish} @ {finish_time.isoformat()}Z")
    print(f"Elapsed time: {duration_slots} slots (~{duration_slots * STEP_MIN} minutes)")
    print(f"τ (slot length): {STEP_MIN} minutes,  t (consecutive 1s): {T_SLOTS}")
    print(f"Distance threshold: ≤ {DIST_KM} km defines a link-up")

if __name__ == "__main__":
    main()
