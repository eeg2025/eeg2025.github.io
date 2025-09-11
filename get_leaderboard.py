import os
import pathlib
import json
from datetime import datetime, timezone
import requests
from urllib.parse import urljoin, urlencode

BASE = os.environ.get("CB_BASE", "https://www.codabench.org")
PK = int(os.environ.get("CB_PK", 9975))
# Read credentials from environment; do NOT hardcode secrets in the repo
USERNAME = os.environ.get("CB_USERNAME", "")
PASSWORD = os.environ.get("CB_PASSWORD", "")
SECRET_KEY = os.environ.get("CB_SECRET_KEY", "")
OUT = pathlib.Path(os.environ.get("CB_OUT", "codabench_results"))
OUT.mkdir(exist_ok=True, parents=True)

def add_secret(url: str) -> str:
    if not SECRET_KEY:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urlencode({'secret_key': SECRET_KEY})}"

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# Fail early if creds are missing (most endpoints require auth)
if not USERNAME or not PASSWORD:
    raise SystemExit(
        "Missing CB_USERNAME/CB_PASSWORD in environment. Set them as CI secrets."
    )

with requests.Session() as s:
    # 1) CSRF from login page
    login_url = urljoin(BASE, "/accounts/login/")
    r = s.get(login_url, timeout=30)
    r.raise_for_status()
    csrftoken = s.cookies.get("csrftoken", "")

    # 2) Try common field names
    headers = {"Referer": login_url}
    payloads = [
        {"username": USERNAME, "password": PASSWORD, "csrfmiddlewaretoken": csrftoken, "next": "/"},
        {"login": USERNAME, "password": PASSWORD, "csrfmiddlewaretoken": csrftoken, "next": "/"},
    ]
    authed = False
    for payload in payloads:
        rp = s.post(login_url, data=payload, headers=headers, timeout=30)
        # Don’t require 2xx here; some sites redirect after login
        # Check auth by hitting a protected-but-readable endpoint:
        test = s.get(urljoin(BASE, f"/api/competitions/{PK}/"), timeout=30)
        if test.status_code == 200:
            authed = True
            comp = test.json()
            break
    if not authed:
        raise SystemExit("Login failed: check BASE/USERNAME/PASSWORD and keep BASE domain consistent.")

    phases = comp.get("phases", [])
    print("Phases:")
    for p in phases:
        print(f"- {p.get('id')} : {p.get('name')}")

    # 3) Try merged JSON (may be forbidden)
    merged_url = add_secret(urljoin(BASE, f"/api/competitions/{PK}/results.json"))
    r = s.get(merged_url, timeout=60)
    merged_saved = None
    if r.status_code == 200:
        merged_saved = OUT / "results_all.json"
        merged_saved.write_bytes(r.content)
        print("Saved merged JSON ->", merged_saved)
    elif r.status_code == 403:
        print("Merged JSON is forbidden for your account; continuing with per-phase endpoints…")
    else:
        print(f"Skipping merged JSON: HTTP {r.status_code}")

    # 4) Per-phase CSV (reliable) + per-phase JSON (if allowed)
    phase_entries = []
    for p in phases:
        pid = p["id"]
        pname = p.get("name")
        # CSV
        csv_url = add_secret(urljoin(BASE, f"/api/competitions/{PK}/results.csv?phase={pid}"))
        rc = s.get(csv_url, timeout=60)
        if rc.ok:
            out_csv = OUT / f"results_phase_{pid}.csv"
            out_csv.write_bytes(rc.content)
            print("Saved CSV ->", out_csv)
        else:
            print(f"CSV HTTP {rc.status_code} for phase {pid}")

        # JSON
        json_url = add_secret(urljoin(BASE, f"/api/competitions/{PK}/results.json?phase={pid}"))
        rj = s.get(json_url, timeout=60)
        if rj.ok:
            out_json = OUT / f"results_phase_{pid}.json"
            out_json.write_bytes(rj.content)
            print("Saved JSON ->", out_json)
            json_path = str(out_json)
        else:
            print(f"JSON HTTP {rj.status_code} for phase {pid}")
            json_path = None

        phase_entries.append({
            "id": pid,
            "name": pname,
            "csv": str(out_csv) if rc.ok else None,
            "json": json_path,
        })

    # 5) Write a lightweight summary with timestamp for easy consumption/commits
    summary = {
        "generated_at_utc": now_utc_iso(),
        "base": BASE,
        "competition_pk": PK,
        "merged_results_json": str(merged_saved) if merged_saved else None,
        "phases": phase_entries,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote summary ->", OUT / "summary.json")
