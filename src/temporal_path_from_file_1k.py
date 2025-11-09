#!/usr/bin/env python3
"""
Temporal shortest path from a downloaded Space-Track JSON file,
capped at ~1,000 satellites, with logging and a vectorized/blocked
connectivity builder to keep things practical.

Edit CONFIG below and run:
  python temporal_path_from_file_1k.py
"""

# ========================= CONFIG (edit me) =========================
JSON_PATH   = "tle_recent_30.json"      # your downloaded JSON (gp/gp_history/tle/tle_latest style)
START_ISO   = "2025-11-08T00:00:00Z"    # UTC start
END_ISO     = "2025-11-08T02:00:00Z"    # UTC end

STEP_MIN    = 2       # slot length τ (minutes)
T_SLOTS     = 2       # consecutive 'up' slots required to traverse an edge
DIST_KM     = 900.0   # link "up" if distance <= DIST_KM

SRC_ID      = 5       # must exist in your filtered/capped set
DST_ID      = 11

OBJECT_LIKE = None    # e.g., "STARLINK" to filter by name substring (case-insensitive)
MAX_SATS    = 500    # cap after filtering; 0 disables the cap
PRINT_SATS  = False   # True: list IDs and names, then exit

# Logging / safety
LOG_EVERY_PROP_STEPS = 5        # log every X time slots during propagation
LARGE_RUN_THRESHOLD  = 500     # warn/refuse if n > threshold (connectivity is O(n^2))
FORCE_LARGE_RUN      = False    # override safety (NOT RECOMMENDED)
# ===================================================================

import json
import math
import time
import itertools
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from dateutil import parser as dtparse
from sgp4.api import Satrec, jday

# --------------------------- logging setup -----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03dZ | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
# force UTC timestamps
logging.Formatter.converter = time.gmtime
log = logging.getLogger("temporal")

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
    log.info("Initialize path matrix …")
    P = initialize_path_matrix(A)
    log.info("Forward pass …")
    forward_pass(A, P)
    log.info("Backward pass …")
    backward_pass(A, P)
    log.info("Selecting shortest-time entries …")
    best = reduce_to_shortest_time(P)
    n = len(A)
    log.info("Reconstructing paths …")
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
    Supports gp, gp_history, tle, tle_latest rows that include TLE_LINE1/2.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array.")
    filt = (object_like or "").lower()
    latest: Dict[int, Tuple[datetime, str, str, str]] = {}
    kept, skipped = 0, 0
    for row in data:
        try:
            name = str(row.get("OBJECT_NAME") or row.get("SATNAME") or "").strip()
            if filt and filt not in name.lower():
                skipped += 1
                continue
            norad = int(row["NORAD_CAT_ID"] if "NORAD_CAT_ID" in row else row["OBJECT_NUMBER"])
            epoch = dtparse.parse(row.get("EPOCH") or row.get("EPOCH_E", ""))
            l1 = row["TLE_LINE1"].strip()
            l2 = row["TLE_LINE2"].strip()
        except Exception:
            skipped += 1
            continue
        prev = latest.get(norad)
        if prev is None or epoch > prev[0]:
            latest[norad] = (epoch, name, l1, l2)
            kept += 1
    log.info(f"Loaded JSON rows: kept {kept}, skipped {skipped}, unique sats: {len(latest)}")
    return {k: (v[1], v[2], v[3]) for k, v in latest.items()}

# ------------------ Vectorized + blocked connectivity builder ----------------
def build_connectivity_strings(
    norad_ids: List[int],
    sat_pos_times: Dict[int, List[np.ndarray]],
    dist_thresh_km: float,
    block: int = 200,  # i/j block size
) -> Tuple[List[List[Optional[EdgeInfo]]], Dict[int, int]]:
    """
    Vectorized + blocked builder:
    - positions: (n, M, 3)
    - per time slot, compute pairwise distances in i/j blocks
    - apply threshold to produce 0/1 connectivity strings per directed edge
    """
    n = len(norad_ids)
    M = len(next(iter(sat_pos_times.values())))
    idx_of = {norad: i for i, norad in enumerate(norad_ids)}

    # pack positions into (n, M, 3)
    P = np.empty((n, M, 3), dtype=np.float64)
    for i, nid in enumerate(norad_ids):
        P[i, :, :] = np.vstack(sat_pos_times[nid])  # (M,3)

    # prepare per-edge connectivity lists
    conn = [[[] for _ in range(n)] for __ in range(n)]

    log.info(f"Building connectivity (n={n}, M={M}, thresh={dist_thresh_km} km, block={block}) …")
    t0 = time.time()
    thr2 = dist_thresh_km * dist_thresh_km

    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        Pi = P[i0:i1]        # (bi, M, 3)
        for j0 in range(0, n, block):
            j1 = min(j0 + block, n)
            Pj = P[j0:j1]    # (bj, M, 3)

            # per slot computation to keep memory in check
            for m in range(M):
                Pi_m = Pi[:, m, :][:, None, :]   # (bi, 1, 3)
                Pj_m = Pj[:, m, :][None, :, :]   # (1, bj, 3)
                d2 = np.sum((Pi_m - Pj_m) ** 2, axis=2)  # (bi, bj)
                up = (d2 <= thr2)

                # append this slot's bit to each directed edge
                for ii in range(i0, i1):
                    row = up[ii - i0]
                    for jj in range(j0, j1):
                        if ii == jj:
                            continue
                        conn[ii][jj].append(1 if row[jj - j0] else 0)

        log.info(f"  connectivity blocks i=[{i0}:{i1}) / {n}")

    log.info(f"Connectivity built in {time.time()-t0:.1f}s")
    # build EdgeInfo grid
    A: List[List[Optional[EdgeInfo]]] = [[None for _ in range(n)] for __ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            A[i][j] = EdgeInfo(connectivity=conn[i][j], t=1)
    return A, idx_of

# --------------------------- Driver -----------------------------------------
def main():
    t0 = dtparse.parse(START_ISO).astimezone(timezone.utc)
    t1 = dtparse.parse(END_ISO).astimezone(timezone.utc)
    if t1 <= t0:
        raise SystemExit("END_ISO must be after START_ISO")
    step = timedelta(minutes=STEP_MIN)

    log.info(f"Loading JSON: {JSON_PATH}")
    latest = load_latest_tles_by_sat(JSON_PATH, OBJECT_LIKE)
    if not latest:
        raise SystemExit("No usable TLEs found in the JSON (after filtering).")

    # select and cap
    norad_ids = sorted(latest.keys())
    log.info(f"Unique satellites before cap: {len(norad_ids)}")
    if MAX_SATS and len(norad_ids) > MAX_SATS:
        norad_ids = norad_ids[:MAX_SATS]
        log.info(f"Capped satellites to first {len(norad_ids)} by NORAD")

    if PRINT_SATS:
        print("NORAD_CAT_ID : OBJECT_NAME")
        for nid in norad_ids[:100]:
            print(f"{nid:<8} : {latest[nid][0]}")
        if len(norad_ids) > 100:
            print(f"... and {len(norad_ids)-100} more")
        return

    if SRC_ID not in norad_ids or DST_ID not in norad_ids:
        raise SystemExit("SRC_ID/DST_ID must be among the filtered/capped set. Set PRINT_SATS=True to inspect.")

    # Build SGP4 propagators
    log.info("Building SGP4 propagators …")
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
    log.info(f"Time grid ready: {M} slots, τ={STEP_MIN} min")

    # Propagation with progress
    log.info(f"Propagating positions for {len(norad_ids)} satellites …")
    sat_pos_times: Dict[int, List[np.ndarray]] = {nid: [] for nid in norad_ids}
    start_prop = time.time()
    for idx_slot, tm in enumerate(times, start=1):
        for nid in norad_ids:
            sat_pos_times[nid].append(eci_position_km(satrecs[nid], tm))
        if idx_slot % max(1, LOG_EVERY_PROP_STEPS) == 0 or idx_slot == M:
            pct = 100.0 * idx_slot / M
            elapsed = time.time() - start_prop
            log.info(f"  propagated slots {idx_slot}/{M} ({pct:.1f}%), elapsed {elapsed:.1f}s")
    log.info(f"Propagation done in {time.time()-start_prop:.1f}s")

    # Safety check for pairwise explosion
    n = len(norad_ids)
    est_pairs = n * (n - 1)
    est_ops = est_pairs * M
    if n > LARGE_RUN_THRESHOLD and not FORCE_LARGE_RUN:
        log.error(
            f"Refusing to build connectivity for n={n} (> {LARGE_RUN_THRESHOLD}). "
            f"Estimated pairwise checks: {est_pairs:,} per slot, total {est_ops:,}."
        )
        log.error("Reduce the problem size: set OBJECT_LIKE (e.g., 'STARLINK'), or lower MAX_SATS, or set FORCE_LARGE_RUN=True (not recommended).")
        return
    else:
        log.info(f"Connectivity estimate -> pairs/slot: {est_pairs:,}, total checks: {est_ops:,}")

    # Build temporal graph (vectorized/blocked)
    A, idx_of = build_connectivity_strings(norad_ids, sat_pos_times, DIST_KM, block=200)
    for i, j in itertools.permutations(range(len(norad_ids)), 2):
        if A[i][j] is not None:
            A[i][j] = EdgeInfo(connectivity=A[i][j].connectivity, t=T_SLOTS)

    # Shortest temporal paths
    log.info("Running temporal shortest-path algorithm …")
    best, best_paths = all_pairs_shortest_time_paths(A)

    si, di = idx_of[SRC_ID], idx_of[DST_ID]
    be = best[si][di]
    if be is None:
        log.warning(f"No temporal path from {SRC_ID} to {DST_ID} in the window.")
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
