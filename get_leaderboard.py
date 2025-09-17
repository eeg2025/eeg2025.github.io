import os
import pathlib
import json
from datetime import datetime, timezone
import requests
from urllib.parse import urljoin, urlencode

DEFAULT_BASE = "https://www.codabench.org"
DEFAULT_PK = 9975  # Competition primary key

def normalize_base(value: str | None) -> str:
    v = (value or "").strip()
    if not v:
        return DEFAULT_BASE
    if not (v.startswith("http://") or v.startswith("https://")):
        v = "https://" + v
    return v.rstrip("/")

def parse_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except Exception:
        return default

BASE = normalize_base(os.environ.get("CB_BASE"))
PK = parse_int(os.environ.get("CB_PK"), DEFAULT_PK)
# Read credentials from environment; do NOT hardcode secrets in the repo
USERNAME = (os.environ.get("CB_USERNAME", "") or "").strip()
PASSWORD = (os.environ.get("CB_PASSWORD", "") or "").strip()
SECRET_KEY = (os.environ.get("CB_SECRET_KEY", "") or "").strip()
OUT = pathlib.Path(os.environ.get("CB_OUT", "codabench_results"))
OUT.mkdir(exist_ok=True, parents=True)

def add_secret(url: str) -> str:
    if not SECRET_KEY:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urlencode({'secret_key': SECRET_KEY})}"

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def prime_secret_cookie(session: requests.Session) -> None:
    """Visit the competition page with `?secret_key=...` to let the server
    set any session flags/cookies that unlock results for secret viewers.
    """
    if not SECRET_KEY:
        return
    try:
        url = add_secret(urljoin(BASE, f"/competitions/{PK}/"))
        r = session.get(url, timeout=30, allow_redirects=True)
        print(f"Primed secret via competition page: HTTP {r.status_code}")
    except Exception as e:
        print(f"Warning: failed to prime secret on competition page: {e}")

def try_login(session: requests.Session) -> bool:
    """Attempt to login; return True if a session is established.
    If CB_USERNAME/PASSWORD are not set, skip and return False.
    """
    if not USERNAME or not PASSWORD:
        print("CB_USERNAME/CB_PASSWORD not set; will try public or secret_key access only.")
        return False

    login_url = urljoin(BASE, "/accounts/login/")
    r = session.get(login_url, timeout=30)
    r.raise_for_status()
    csrftoken = session.cookies.get("csrftoken", "")
    headers = {"Referer": login_url, "X-CSRFToken": csrftoken}
    payloads = [
        {"username": USERNAME, "password": PASSWORD, "csrfmiddlewaretoken": csrftoken, "next": "/"},
        {"login": USERNAME, "password": PASSWORD, "csrfmiddlewaretoken": csrftoken, "next": "/"},
    ]
    for payload in payloads:
        rp = session.post(login_url, data=payload, headers=headers, timeout=30, allow_redirects=True)
        if session.cookies.get("sessionid"):
            print("Login successful (session established).")
            return True
    print("Login attempt did not establish a session; proceeding without auth.")
    return False

with requests.Session() as s:
    print(f"Using BASE={BASE}, PK={PK}")
    print("Auth present:", bool(USERNAME and PASSWORD))
    print("Secret key present:", bool(SECRET_KEY))
    # Prime secret access first (if provided) — some deployments require
    # hitting the page with the secret to set a session flag used by APIs
    prime_secret_cookie(s)
    try_login(s)
    print("Cookies set:", [c.name for c in s.cookies])
    # Public competition details endpoint
    test = s.get(urljoin(BASE, f"/api/competitions/{PK}/"), timeout=30)
    test.raise_for_status()
    comp = test.json()

    phases = comp.get("phases", [])
    print("Phases:")
    for p in phases:
        print(f"- {p.get('id')} : {p.get('name')}")

    # 3) Try merged JSON (may be forbidden without secret/admin)
    merged_url = add_secret(urljoin(BASE, f"/api/competitions/{PK}/results.json"))
    merged_referer = add_secret(urljoin(BASE, f"/competitions/{PK}/"))
    r = s.get(merged_url, timeout=60, headers={"Referer": merged_referer})
    merged_saved = None
    if r.status_code == 200:
        merged_saved = OUT / "results_all.json"
        merged_saved.write_bytes(r.content)
        print("Saved merged JSON ->", merged_saved)
    elif r.status_code == 403:
        msg = "Merged JSON forbidden (need secret_key or admin)."
        if not SECRET_KEY:
            msg += " Set CB_SECRET_KEY in CI for access."
        print(msg)
    else:
        print(f"Skipping merged JSON: HTTP {r.status_code}")

    # 4) Per-phase CSV (reliable) + per-phase JSON (if allowed)
    phase_entries = []
    for p in phases:
        pid = p["id"]
        pname = p.get("name")
        # CSV
        csv_url = add_secret(urljoin(BASE, f"/api/competitions/{PK}/results.csv?phase={pid}"))
        referer = add_secret(urljoin(BASE, f"/competitions/{PK}/"))
        out_csv = OUT / f"results_phase_{pid}.csv"
        rc = s.get(csv_url, timeout=60, headers={"Referer": referer})
        if rc.ok:
            out_csv.write_bytes(rc.content)
            csv_path = str(out_csv)
            print("Saved CSV ->", out_csv)
        else:
            csv_path = None
            if rc.status_code == 403 and not SECRET_KEY:
                print(f"CSV HTTP 403 for phase {pid} (set CB_SECRET_KEY for access)")
            else:
                print(f"CSV HTTP {rc.status_code} for phase {pid}")

        # JSON
        json_url = add_secret(urljoin(BASE, f"/api/competitions/{PK}/results.json?phase={pid}"))
        rj = s.get(json_url, timeout=60, headers={"Referer": referer})
        if rj.ok:
            out_json = OUT / f"results_phase_{pid}.json"
            out_json.write_bytes(rj.content)
            print("Saved JSON ->", out_json)
            json_path = str(out_json)
        else:
            if rj.status_code == 403 and not SECRET_KEY:
                print(f"JSON HTTP 403 for phase {pid} (set CB_SECRET_KEY for access)")
            else:
                print(f"JSON HTTP {rj.status_code} for phase {pid}")
            json_path = None

        # Leaderboard metadata (includes submission timestamps)
        leaderboard_path = None
        leaderboard_url = add_secret(urljoin(BASE, f"/api/phases/{pid}/get_leaderboard/"))
        try:
            rl = s.get(
                leaderboard_url,
                timeout=60,
                headers={
                    "Referer": referer,
                    "Accept": "application/json",
                },
            )
        except Exception as exc:
            print(f"Leaderboard fetch error for phase {pid}: {exc}")
        else:
            if rl.ok:
                out_leaderboard = OUT / f"leaderboard_phase_{pid}.json"
                out_leaderboard.write_bytes(rl.content)
                leaderboard_path = str(out_leaderboard)
                print("Saved leaderboard ->", out_leaderboard)
            else:
                if rl.status_code == 403 and not SECRET_KEY:
                    print(f"Leaderboard HTTP 403 for phase {pid} (set CB_SECRET_KEY for access)")
                else:
                    print(f"Leaderboard HTTP {rl.status_code} for phase {pid}")

        phase_entries.append({
            "id": pid,
            "name": pname,
            "csv": csv_path,
            "json": json_path,
            "leaderboard": leaderboard_path,
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
