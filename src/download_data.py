#!/usr/bin/env python3
import os
import sys
import time
import argparse
import textwrap
from typing import List, Tuple, Optional
from urllib.parse import quote
import requests

"""
Space-Track.org downloader, aligned with the REST rules in the official docs.

Features:
- Auth (session cookie) via username/password
- Build REST URLs for classes like gp, gp_history, tle, tle_latest, satcat, etc.
- Add REST predicates like EPOCH/>now-30, NORAD_CAT_ID/25544, DECAY/null-val, etc.
- Orderby, limit, favorites, distinct, metadata
- Output formats: json, csv, xml, html, tle, 3le, kvn (per docs)
- Saves to file, prints a small summary
- Basic retry/backoff for throttling (HTTP 429/5xx)

Usage examples:
  python spacetrack_downloader.py --class gp --pred EPOCH/>now-30 --format json --out gp_now30.json
  python spacetrack_downloader.py --class gp_history --pred NORAD_CAT_ID/25544 --orderby "EPOCH desc" --limit 22 --format tle --out iss_22_latest.tle

Docs keys reflected:
- Controllers: basicspacedata (default), expandedspacedata, publicfiles, eventsdata
- Action: query (default), modeldef
- REST Predicates: /predicates/, /metadata/true, /limit/x(,x?), /orderby/PRED [asc|desc], /distinct/true, /format/xxx, /favorites/...
"""

BASE_URL = "https://www.space-track.org"
LOGIN_PATH = "/ajaxauth/login"
DEFAULT_CONTROLLER = "basicspacedata"
DEFAULT_ACTION = "query"

# Per “API Use Guidelines”: keep it polite.
MAX_RETRIES = 5
BACKOFF_SECS = 5

def get_session(username: str, password: str) -> requests.Session:
    s = requests.Session()
    resp = s.post(
        BASE_URL + LOGIN_PATH,
        data={"identity": username, "password": password},
        timeout=30,
    )
    # Space-Track returns HTML; a 200 with a session cookie usually indicates success.
    try:
        resp.raise_for_status()
    except Exception as e:
        raise SystemExit(f"Login failed: HTTP {resp.status_code} {e}")
    if "Set-Cookie" not in str(resp.headers) and "Session" not in resp.text:
        # Heuristic; login HTML often contains "Session"
        # Still allow it to proceed, but warn.
        print("[warn] Login response did not include obvious session markers. Continuing...", file=sys.stderr)
    return s

def build_url(
    controller: str,
    action: str,
    class_name: str,
    # Predicates as already-formed "PREDICATE/VALUE" segments, e.g. "EPOCH/>now-30"
    predicate_pairs: List[str],
    # Optional standard REST predicates
    out_format: Optional[str] = None,
    orderby: Optional[str] = None,
    limit: Optional[str] = None,       # e.g. "10" or "10,5"
    distinct: bool = False,
    favorites: Optional[str] = None,   # e.g. "Amateur" or "my_favorites"
    predicates_list: Optional[str] = None,  # e.g. "all" or "TLE_LINE1,TLE_LINE2"
    metadata_true: bool = False,
    emptyresult_show: bool = False,
    recursive: Optional[bool] = None
) -> str:
    """
    Build a RESTful Space-Track URL with path segments, escaping path components.
    We follow the doc's style: /basicspacedata/query/class/gp/EPOCH/%3Enow-30/orderby/NORAD_CAT_ID/format/json
    """
    parts = [BASE_URL, controller, action, "class", class_name]

    # Append user-supplied predicate pairs in order
    # Each comes like "EPOCH/>now-30" or "NORAD_CAT_ID/25544"
    for pair in predicate_pairs:
        # Split once to allow embedded '/' in value, if any
        if "/" not in pair:
            # If user gave raw "EPOCH>now-30" style by mistake, try to be helpful
            raise SystemExit(f"Bad --pred '{pair}'. Use 'FIELD/VALUE' (e.g. EPOCH/>now-30).")
        key, value = pair.split("/", 1)
        parts.append(key)
        parts.append(value)

    # Optional REST predicates
    if predicates_list:  # e.g., "all" or "OBJECT_NAME,TLE_LINE1,TLE_LINE2"
        parts.extend(["predicates", predicates_list])
    if metadata_true:
        parts.extend(["metadata", "true"])
    if limit:
        parts.extend(["limit", limit])
    if orderby:
        parts.extend(["orderby", orderby])
    if distinct:
        parts.extend(["distinct", "true"])
    if favorites:
        parts.extend(["favorites", favorites])
    if recursive is not None:
        parts.extend(["recursive", "true" if recursive else "false"])
    if out_format:
        parts.extend(["format", out_format])
    if emptyresult_show:
        parts.extend(["emptyresult", "show"])

    # Percent-encode each path segment properly
    # Keep slashes that separate segments; encode segment content only.
    encoded = []
    for i, seg in enumerate(parts):
        # Leave the "https://" + host part intact for the first element
        if i == 0 and seg.startswith("http"):
            encoded.append(seg.rstrip("/"))
        else:
            # encode spaces, operators, commas, etc.
            encoded.append(quote(seg, safe=""))

    # Join into a proper URL path
    # e.g., https://www.space-track.org/basicspacedata/query/class/gp/EPOCH/%3Enow-30/format/json
    url = "/".join(encoded)
    # Space-Track examples often end with a trailing slash; not required but harmless
    if not url.endswith("/"):
        url += "/"
    return url

def polite_get(session: requests.Session, url: str) -> requests.Response:
    """
    GET with simple retry/backoff for throttling (429) or transient 5xx.
    Respectful of the guideline to limit frequency.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        resp = session.get(url, timeout=120)
        if resp.status_code < 400:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = BACKOFF_SECS * attempt
            print(f"[warn] HTTP {resp.status_code} on attempt {attempt}. Backing off {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        # Other client errors: show and stop
        try:
            msg = resp.json()
        except Exception:
            msg = resp.text
        raise SystemExit(f"Request failed: HTTP {resp.status_code} - {msg}")
    raise SystemExit("Max retries exceeded.")

def save_bytes(content: bytes, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(content)

def main():
    ap = argparse.ArgumentParser(
        description="Download data from Space-Track.org via REST API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Newest propagable sets last 30 days (recommended 'gp' class), JSON:
          spacetrack_downloader.py --class gp --pred EPOCH/>now-30 --pred decay_date/null-val --orderby norad_cat_id --format json --out gp_now30.json

          # Historical TLEs (gp_history) for ISS as TLE text:
          spacetrack_downloader.py --class gp_history --pred NORAD_CAT_ID/25544 --orderby "EPOCH desc" --limit 22 --format tle --out iss_22_latest.tle

          # Old-style tle_latest, one per object:
          spacetrack_downloader.py --class tle_latest --pred ORDINAL/1 --format json --out tle_latest.json

          # SATCAT objects launched in last 7 days (HTML default->JSON if omitted):
          spacetrack_downloader.py --class satcat --pred LAUNCH/>now-7 --pred CURRENT/Y --orderby "LAUNCH DESC" --format html --out satcat_recent.html
        """)
    )
    ap.add_argument("--user", default=os.getenv("SPACETRACK_USER"), help="Space-Track username (or set SPACETRACK_USER)")
    ap.add_argument("--pass", dest="password", default=os.getenv("SPACETRACK_PASS"), help="Space-Track password (or set SPACETRACK_PASS)")
    ap.add_argument("--controller", default=DEFAULT_CONTROLLER, help="Controller (basicspacedata|expandedspacedata|publicfiles|eventsdata)")
    ap.add_argument("--action", default=DEFAULT_ACTION, help="Action (query|modeldef)")
    ap.add_argument("--class", dest="class_name", required=True, help="API class, e.g., gp, gp_history, tle, tle_latest, satcat, boxscore, decay, tip, etc.")
    ap.add_argument("--pred", action="append", default=[], help="REST predicate pair like FIELD/VALUE (repeatable). Examples: EPOCH/>now-30  NORAD_CAT_ID/25544  DECAY/null-val")
    ap.add_argument("--predicates-list", help="Value for /predicates/, e.g. 'all' or 'OBJECT_NAME,TLE_LINE1,TLE_LINE2'")
    ap.add_argument("--orderby", help='e.g. "EPOCH desc" or "NORAD_CAT_ID"')
    ap.add_argument("--limit", help='e.g. "10" or "10,5"')
    ap.add_argument("--distinct", action="store_true", help="Add /distinct/true")
    ap.add_argument("--favorites", help="Favorites list name, e.g. Amateur, my_favorites, All")
    ap.add_argument("--metadata", action="store_true", help="Add /metadata/true (ignored by tle/3le/csv per docs)")
    ap.add_argument("--emptyresult-show", action="store_true", help="Add /emptyresult/show (shows a message if no results)")
    ap.add_argument("--recursive", choices=["true", "false"], help="Only relevant for API folder downloads (publicfiles)")
    ap.add_argument("--format", dest="out_format", default="json", help="json|csv|xml|html|tle|3le|kvn")
    ap.add_argument("--out", required=True, help="Output file path to save response")

    args = ap.parse_args()

    if not args.user or not args.password:
        raise SystemExit("Set credentials with --user/--pass or environment variables SPACETRACK_USER / SPACETRACK_PASS.")

    recursive_bool = None
    if args.recursive is not None:
        recursive_bool = True if args.recursive == "true" else False

    # Build URL
    url = build_url(
        controller=args.controller,
        action=args.action,
        class_name=args.class_name,
        predicate_pairs=args.pred,
        out_format=args.out_format,
        orderby=args.orderby,
        limit=args.limit,
        distinct=args.distinct,
        favorites=args.favorites,
        predicates_list=args.predicates_list,
        metadata_true=args.metadata,
        emptyresult_show=args.emptyresult_show,
        recursive=recursive_bool
    )

    print(f"[info] Request URL:\n{url}\n", file=sys.stderr)

    # Login and fetch
    session = get_session(args.user, args.password)
    resp = polite_get(session, url)

    # Save
    save_bytes(resp.content, args.out)

    # Tiny summary
    ct = resp.headers.get("Content-Type", "")
    size_kb = len(resp.content) / 1024.0
    print(f"[ok] Saved {size_kb:.1f} KB ({ct}) to: {args.out}")

if __name__ == "__main__":
    main()
