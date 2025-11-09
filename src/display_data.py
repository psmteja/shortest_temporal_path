import json
from pathlib import Path

FNAME = "tle_recent_30.json"   # change if needed
N = 5                          # how many to show

def safe_get(d, key, default=""):
    return d.get(key, default)

def main():
    p = Path(FNAME)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array from Space-Track.")

    print(f"Total records: {len(data)}; showing top {min(N, len(data))}\n")

    # Common fields returned by gp/tle endpoints
    preferred_fields = [
        "OBJECT_NAME", "NORAD_CAT_ID", "EPOCH",
        "TLE_LINE0", "TLE_LINE1", "TLE_LINE2"
    ]

    for i, row in enumerate(data[:N], start=1):
        print(f"[{i}] ------------------------------")
        # print preferred fields first (if they exist)
        for k in preferred_fields:
            if k in row:
                print(f"{k:13}: {row[k]}")
        # print a compact fallback if nothing matched
        if not any(k in row for k in preferred_fields):
            # show first 6 key:value pairs as a fallback preview
            for k in list(row.keys())[:6]:
                print(f"{k:13}: {row[k]}")
        print()

if __name__ == "__main__":
    main()
