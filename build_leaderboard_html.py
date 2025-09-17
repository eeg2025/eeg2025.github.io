import os
import re
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent.resolve()
CB_DIR = Path(os.environ.get("CB_OUT", ROOT / "codabench_results"))
DEST = ROOT / "_includes" / "leaderboard_generated.html"

# Optional overrides via environment
OVR_REGEX = os.environ.get("LB_METRIC_REGEX")
OVR_LABEL = os.environ.get("LB_LABEL")
OVR_DIRECTION = os.environ.get("LB_DIRECTION")  # 'lower' or 'higher'
PHASE_REGEX = os.environ.get("LB_PHASE_REGEX", r"(?i)(warm\s*up|warmup|development|dev|phase\s*1)")
X_START = os.environ.get("LB_X_START")  # e.g., '2025-09-01' (UTC)
X_END = os.environ.get("LB_X_END")      # e.g., '2025-10-03' (UTC)
ACCENT = os.environ.get("LB_ACCENT", "#F2994A")
ACCENT_DARK = os.environ.get("LB_ACCENT_DARK", "#D97E2E")
ACCENT_LIGHT = os.environ.get("LB_ACCENT_LIGHT", "rgba(242,153,74,0.30)")

CH2_COLOR = os.environ.get("LB_CHALLENGE2_COLOR", "#2D9CDB")
CH2_DARK = os.environ.get("LB_CHALLENGE2_DARK", "#1B75BB")
CH2_LIGHT = os.environ.get("LB_CHALLENGE2_LIGHT", "rgba(45,156,219,0.28)")

COMBINED_COLOR = os.environ.get("LB_COMBINED_COLOR", "#9B51E0")
COMBINED_DARK = os.environ.get("LB_COMBINED_DARK", "#6F3AB2")
COMBINED_LIGHT = os.environ.get("LB_COMBINED_LIGHT", "rgba(155,81,224,0.28)")

TITLE_PREFIX = os.environ.get("LB_TITLE_PREFIX", "Leaderboard")

# Track phase names seen so we can render a figure even if a phase has no data
PHASE_NAMES_SEEN: set[str] = set()

SUBMISSION_ID_RE = re.compile(r"-(\d+)$")

METRIC_CONFIG: list[dict[str, Any]] = [
    {
        "key": "challenge1",
        "title": "Challenge 1",
        "label": "Challenge 1 Score",
        "regex": r"(?i)challenge\s*1",
        "direction": "lower",
        "line_color": ACCENT,
        "line_dark": ACCENT_DARK,
        "scatter_color": ACCENT_LIGHT,
        "order": 0,
        "best_text_position": "top center",
    },
    {
        "key": "challenge2",
        "title": "Challenge 2",
        "label": "Challenge 2 Score",
        "regex": r"(?i)challenge\s*2",
        "direction": "lower",
        "line_color": CH2_COLOR,
        "line_dark": CH2_DARK,
        "scatter_color": CH2_LIGHT,
        "order": 1,
        "best_text_position": "bottom center",
    },
    {
        "key": "overall",
        "title": "Combined",
        "label": "Combined Score",
        "regex": r"(?i)(normalized|normalised)\s*mse|\bnmse\b|overall score$",
        "direction": "lower",
        "line_color": COMBINED_COLOR,
        "line_dark": COMBINED_DARK,
        "scatter_color": COMBINED_LIGHT,
        "order": 2,
        "best_text_position": "middle right",
    },
]

METRIC_STYLE_LOOKUP = {cfg["key"]: cfg for cfg in METRIC_CONFIG}


def ensure_metric_style(metric: dict[str, Any]) -> dict[str, Any]:
    key = metric.get("key")
    if not key:
        return metric
    if key not in METRIC_STYLE_LOOKUP:
        METRIC_STYLE_LOOKUP[key] = {
            k: v for k, v in metric.items() if k not in {"source"}
        }
    return METRIC_STYLE_LOOKUP[key]


def custom_metric_def(source: str, direction: str | None, label: str | None) -> dict[str, Any]:
    lbl = label or str(source)
    dirn = direction or (OVR_DIRECTION or "lower")
    slug = re.sub(r"\W+", "_", str(source).strip().lower()) or "custom"
    return {
        "key": f"custom_{slug}",
        "title": lbl,
        "label": lbl,
        "regex": OVR_REGEX or "",
        "direction": dirn,
        "line_color": ACCENT,
        "line_dark": ACCENT_DARK,
        "scatter_color": ACCENT_LIGHT,
        "order": 99,
        "best_text_position": "top center",
        "source": source,
    }


def style_for_metric(metric_key: str, label: str, direction: str) -> dict[str, Any]:
    style = METRIC_STYLE_LOOKUP.get(metric_key)
    if style:
        return style
    style = {
        "key": metric_key,
        "title": label,
        "label": label,
        "direction": direction,
        "line_color": ACCENT,
        "line_dark": ACCENT_DARK,
        "scatter_color": ACCENT_LIGHT,
        "order": 99,
    }
    METRIC_STYLE_LOOKUP[metric_key] = style
    return style


def match_metric_configs(names: list[str]) -> list[dict[str, Any]]:
    if not names:
        return []
    normalized = [str(n) for n in names]
    matches: list[dict[str, Any]] = []
    for cfg in sorted(METRIC_CONFIG, key=lambda c: c.get("order", 0)):
        for name in normalized:
            if re.search(cfg["regex"], name):
                matches.append(dict(cfg, source=name))
                break
    if not matches:
        return []
    if OVR_REGEX:
        try:
            rx = re.compile(OVR_REGEX)
        except re.error:
            rx = None
        else:
            filtered = [m for m in matches if rx.search(m["source"]) or rx.search(str(m.get("label", ""))) or rx.search(str(m.get("title", "")))]
            if filtered:
                matches = filtered
    return matches


def fallback_metric_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    cols = list(df.columns)
    if not cols:
        return []
    ignore = {c.lower() for c in ["Username", "fact_sheet_answers", "Team", "team_name", "Date", "date"]}
    metric_cols = [c for c in cols if str(c).strip().lower() not in ignore]
    if not metric_cols:
        metric_cols = cols
    direction = OVR_DIRECTION or "lower"
    label = OVR_LABEL
    column = None

    if OVR_REGEX:
        try:
            rx = re.compile(OVR_REGEX)
        except re.error:
            rx = None
        if rx:
            for c in metric_cols:
                if rx.search(str(c)):
                    column = c
                    break
    if column is None:
        # reuse existing heuristics
        col, dir_guess, label_guess = select_metric_column(df)
        column = col
        if dir_guess:
            direction = dir_guess
        if label is None:
            label = label_guess
    if column is None:
        return []
    return [custom_metric_def(column, direction, label)]

def parse_time(s: str | None, fallback: float) -> datetime:
    if s:
        try:
            if s.endswith("Z"):
                for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
                    try:
                        dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                        return dt.astimezone(timezone.utc)
                    except ValueError:
                        continue
                dt = datetime.fromisoformat(s[:-1] + "+00:00").astimezone(timezone.utc)
            else:
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    return datetime.fromtimestamp(fallback, tz=timezone.utc)


def discover_summaries(base: Path) -> list[Path]:
    if not base.exists():
        return []
    paths = list(base.glob("summary.json")) + list(base.rglob("summary.json"))
    seen, uniq = set(), []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def list_phase_entries(summary: dict) -> list[dict]:
    """Return a list of dicts with keys: name, csv, json for each phase.
    Paths are absolute where possible; missing or tiny CSVs are kept as None.
    """
    out: list[dict] = []
    for p in summary.get("phases") or []:
        name = p.get("name") or ""
        try:
            if name:
                PHASE_NAMES_SEEN.add(str(name))
        except Exception:
            pass
        csv = p.get("csv")
        j = p.get("json")
        csv_path = None
        if csv:
            cp = Path(csv)
            if not cp.is_absolute():
                cp = (ROOT / cp).resolve()
            if cp.exists() and cp.stat().st_size >= 10:
                csv_path = cp
        json_path = None
        if j:
            jp = Path(j)
            if not jp.is_absolute():
                jp = (ROOT / jp).resolve()
            if jp.exists() and jp.stat().st_size >= 10:
                json_path = jp
        leader = p.get("leaderboard")
        leaderboard_path = None
        if leader:
            lp = Path(leader)
            if not lp.is_absolute():
                lp = (ROOT / lp).resolve()
            if lp.exists() and lp.stat().st_size >= 10:
                leaderboard_path = lp
        out.append({"name": name, "csv": csv_path, "json": json_path, "leaderboard": leaderboard_path})
    # If none, try merged file fallback (rare)
    if not out:
        merged = summary.get("merged_results_json")
        if merged:
            mp = Path(merged)
            if not mp.is_absolute():
                mp = (ROOT / mp).resolve()
            if mp.exists():
                out.append({"name": "", "csv": None, "json": mp, "leaderboard": None})
    return out


def split_user_and_submission(raw: str | None) -> tuple[str, str | None]:
    if raw is None:
        return "", None
    text = str(raw).strip()
    match = SUBMISSION_ID_RE.search(text)
    if match:
        return text[:match.start()], match.group(1)
    return text, None


def load_leaderboard_meta(path: Path | None) -> dict[str, Any]:
    if not path:
        return {"by_id": {}, "by_owner": {}}
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return {"by_id": {}, "by_owner": {}}

    id_map: dict[str, dict[str, Any]] = {}
    owner_map: dict[str, list[dict[str, Any]]] = {}

    def record(entry: dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            return
        sid_raw = entry.get("id") or entry.get("submission_id") or entry.get("pk")
        if sid_raw is None:
            return
        sid = str(sid_raw)
        created = entry.get("created_when") or entry.get("created_at") or entry.get("submitted_at") or entry.get("created")
        created_dt = None
        if created:
            try:
                created_dt = parse_time(str(created), 0)
            except Exception:
                created_dt = None
        owner = entry.get("owner") or entry.get("participant") or entry.get("username") or entry.get("user")
        info = {
            "submission_id": sid,
            "created": created,
            "created_dt": created_dt,
            "owner": owner,
        }
        id_map[sid] = info
        owner_key = (str(owner).strip().lower()) if owner else ""
        if owner_key:
            owner_map.setdefault(owner_key, []).append(info)

    if isinstance(data, dict):
        submissions = data.get("submissions")
        if isinstance(submissions, list):
            for entry in submissions:
                record(entry)
        leaderboards = data.get("leaderboards")
        if isinstance(leaderboards, list):
            for leaderboard in leaderboards:
                if isinstance(leaderboard, dict):
                    for entry in leaderboard.get("submissions", []) or []:
                        record(entry)

    sentinel = datetime.max.replace(tzinfo=timezone.utc)
    for key, infos in owner_map.items():
        infos.sort(key=lambda info: info.get("created_dt") or sentinel)

    return {"by_id": id_map, "by_owner": owner_map}


def resolve_submission_times(df: pd.DataFrame, leaderboard_meta: dict[str, Any], fallback: datetime) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")

    fallback_dt = fallback if isinstance(fallback, datetime) else datetime.fromtimestamp(float(fallback), tz=timezone.utc)
    fallback_ts = fallback_dt.timestamp()

    id_map = leaderboard_meta.get("by_id", {}) if leaderboard_meta else {}
    owner_map = leaderboard_meta.get("by_owner", {}) if leaderboard_meta else {}

    resolved: list[datetime] = []
    users = df.get("user") if "user" in df.columns else pd.Series([None] * len(df))

    for submission_id, user in zip(df.get("submission_id", []), users):
        dt = None
        if submission_id and id_map:
            info = id_map.get(str(submission_id))
            if info:
                dt = info.get("created_dt")
                if dt is None and info.get("created"):
                    try:
                        dt = parse_time(str(info["created"]), fallback_ts)
                    except Exception:
                        dt = None
        if dt is None and owner_map and user:
            owner_key = str(user).strip().lower()
            if owner_key:
                infos = owner_map.get(owner_key)
                if infos:
                    info = infos[0]
                    dt = info.get("created_dt")
                    if dt is None and info.get("created"):
                        try:
                            dt = parse_time(str(info["created"]), fallback_ts)
                        except Exception:
                            dt = None
        if dt is None:
            dt = fallback_dt
        resolved.append(dt)

    return pd.to_datetime(resolved)


def compute_axis_range(values: pd.Series, pad_ratio: float = 0.12, sigma_cap: float = 4.0) -> tuple[list[float], float] | None:
    if values is None:
        return None
    clean = pd.Series(values).dropna()
    clean = clean[clean > 0]
    if clean.empty:
        return None
    min_val = float(clean.min())
    max_val = float(clean.max())
    if min_val <= 0:
        min_val = min(filter(lambda v: v > 0, clean.tolist()))
    if min_val == max_val:
        pad = min_val * pad_ratio if min_val != 0 else pad_ratio
        lower = min_val - pad
        if lower <= 0:
            lower = min_val * 0.9 if min_val > 0 else 0.1
        upper = min_val + pad
        if upper <= lower:
            upper = lower * 1.05
        return [lower, upper], upper

    q_low = float(clean.quantile(0.05))
    q_high = float(clean.quantile(0.95))
    span = q_high - q_low
    if span <= 0:
        span = max_val - min_val
    pad = span * pad_ratio
    lower = max(min_val * 0.9, q_low - pad)
    upper = min(max_val * 1.08, q_high + pad)
    mean_val = float(clean.mean())
    std_val = float(clean.std(ddof=0))
    if std_val > 0:
        cap_candidate = mean_val + sigma_cap * std_val
        upper = min(upper, cap_candidate)
    upper = min(upper, max_val)
    if upper <= lower:
        upper = lower * 1.05
    return [lower, upper], upper


def phase_xrange(phase_name: str | None) -> tuple[str, str] | None:
    """Return a hard-coded x-axis date range based on phase name.
    Warmup: 2025-09-01 → 2025-10-03
    Final:  2025-10-04 → 2025-11-03
    Else:   LB_X_START/LB_X_END if provided; otherwise None.
    """
    nm = (phase_name or "").lower()
    if "warm" in nm or "phase 1" in nm or "development" in nm or "dev" in nm:
        return ("2025-09-01", "2025-10-03")
    if "final" in nm:
        return ("2025-10-04", "2025-11-03")
    if X_START and X_END:
        return (X_START, X_END)
    return None


def select_metric_column(df: pd.DataFrame) -> tuple[str | None, str, str]:
    cols = list(df.columns)
    # Ignore typical non-metric columns
    ignore = {c.lower() for c in ["Username", "fact_sheet_answers", "Team", "team_name", "Date", "date"]}
    metric_cols = [c for c in cols if str(c).strip().lower() not in ignore]

    # Environment override first
    if OVR_REGEX:
        try:
            rx = re.compile(OVR_REGEX)
            for c in metric_cols:
                if rx.search(str(c)):
                    return c, (OVR_DIRECTION or "lower"), (OVR_LABEL or "Normalize MSE")
        except re.error:
            pass

    # Prefer Challenge 1 metrics when present
    for c in metric_cols:
        if re.search(r"challenge\s*1", str(c), re.I):
            return c, "lower", "Challenge 1 Score"

    # Normalized MSE / NMSE
    for c in metric_cols:
        if re.search(r"(normalized|normalised)\s*mse\b", str(c), re.I) or re.search(r"\bnmse\b", str(c), re.I):
            return c, "lower", "Normalize MSE"
    # MSE
    for c in metric_cols:
        if re.search(r"\bmse\b", str(c), re.I):
            return c, "lower", "MSE"
    # overall score
    for c in metric_cols:
        if re.search(r"overall score$", str(c), re.I):
            return c, "higher", "Score"
    # First numeric column
    for c in metric_cols:
        try:
            pd.to_numeric(df[c])
            return c, (OVR_DIRECTION or "higher"), (OVR_LABEL if OVR_DIRECTION else "Score")
        except Exception:
            continue
    return None, "higher", "Score"


def metric_from_keys(keys: list[str]) -> tuple[str | None, str, str]:
    # Environment override
    if OVR_REGEX:
        try:
            rx = re.compile(OVR_REGEX)
            for k in keys:
                if rx.search(str(k)):
                    return k, (OVR_DIRECTION or "lower"), (OVR_LABEL or "Normalize MSE")
        except re.error:
            pass
    for k in keys:
        if re.search(r"challenge\s*1", str(k), re.I):
            return k, "lower", "Challenge 1 Score"
    for k in keys:
        if re.search(r"(normalized|normalised)\s*mse\b", str(k), re.I) or re.search(r"\bnmse\b", str(k), re.I):
            return k, "lower", "Normalize MSE"
    for k in keys:
        if re.search(r"\bmse\b", str(k), re.I):
            return k, "lower", "MSE"
    for k in keys:
        if re.search(r"overall score$", str(k), re.I):
            return k, "higher", "Score"
    return (keys[0] if keys else None), (OVR_DIRECTION or "higher"), (OVR_LABEL if OVR_DIRECTION else "Score")


def find_result_json(summary: dict, summary_path: Path) -> Path | None:
    merged = summary.get("merged_results_json")
    if merged:
        p = Path(merged)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if p.exists():
            return p
    phases = summary.get("phases") or []
    for p in phases:
        j = p.get("json")
        if not j:
            continue
        pj = Path(j)
        if not pj.is_absolute():
            pj = (ROOT / pj).resolve()
        if pj.exists():
            return pj
    candidate = summary_path.parent / "results_all.json"
    return candidate if candidate.exists() else None


def pick_phase_block(results: dict) -> tuple[str, dict]:
    if not isinstance(results, dict):
        return "N/A", {}
    keys = list(results.keys())
    if not keys:
        return "N/A", {}
    for k in keys:
        if re.search(r"final", k, re.I) and isinstance(results.get(k), dict) and results[k]:
            return k, results[k]
    for k in keys:
        if isinstance(results.get(k), dict) and results[k]:
            return k, results[k]
    return keys[0], results.get(keys[0], {})


def load_dataframe() -> pd.DataFrame:
    rows = []
    for sp in discover_summaries(CB_DIR):
        try:
            summary = json.loads(sp.read_text())
        except Exception:
            continue
        when = parse_time(summary.get("generated_at_utc") or summary.get("generated_at"), sp.stat().st_mtime)
        # Iterate all phases present in this snapshot
        for ph in list_phase_entries(summary):
            phase_name = ph.get("name") or ""
            used_any = False
            leaderboard_meta = load_leaderboard_meta(ph.get("leaderboard"))
            # CSV first
            csv_path = ph.get("csv")
            if csv_path:
                try:
                    df = pd.read_csv(csv_path)
                    # Username column
                    user_col = None
                    for cand in ["Username", "user", "User", "team", "Team", "participant", "Participant"]:
                        if cand in df.columns:
                            user_col = cand
                            break
                    if user_col is None:
                        user_col = df.columns[0]
                    metric_defs = match_metric_configs(list(df.columns))
                    if not metric_defs:
                        metric_defs = fallback_metric_from_df(df)
                    for metric_def in metric_defs:
                        col_name = metric_def.get("source")
                        if not col_name or col_name not in df.columns:
                            continue
                        ensure_metric_style(metric_def)
                        tmp = df[[user_col, col_name]].copy()
                        tmp = tmp.rename(columns={user_col: "raw_user", col_name: "score"})
                        tmp["raw_user"] = tmp["raw_user"].astype(str)
                        splits = tmp["raw_user"].map(split_user_and_submission)
                        tmp["user"] = splits.str[0]
                        tmp["submission_id"] = splits.str[1]
                        tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce")
                        tmp = tmp.dropna(subset=["score"]).reset_index(drop=True)
                        if tmp.empty:
                            continue
                        direction = metric_def.get("direction") or "lower"
                        tmp["when"] = resolve_submission_times(tmp, leaderboard_meta, when)
                        tmp = tmp.drop(columns=["raw_user"], errors="ignore")
                        tmp["label"] = metric_def.get("label", metric_def.get("title", "Score"))
                        tmp["metric_key"] = metric_def.get("key")
                        tmp["metric_label"] = metric_def.get("label", metric_def.get("title", "Score"))
                        tmp["direction"] = direction
                        tmp["phase"] = phase_name
                        idx = tmp["score"].idxmax() if direction == "higher" else tmp["score"].idxmin()
                        tmp["is_best"] = False
                        if idx is not None and idx in tmp.index:
                            tmp.loc[idx, "is_best"] = True
                        rows.append(tmp)
                        used_any = True
                except Exception:
                    pass
            if used_any:
                continue
            # JSON fallback for this phase
            jpath = ph.get("json")
            if not jpath:
                continue
            try:
                results = json.loads(jpath.read_text())
            except Exception:
                continue
            # Find the block for this phase by name if possible
            entries = None
            if isinstance(results, dict):
                for k, v in results.items():
                    if phase_name and phase_name.lower() in k.lower():
                        entries = v
                        break
                if entries is None:
                    # fallback to first non-empty
                    for k, v in results.items():
                        if isinstance(v, dict) and v:
                            entries = v
                            break
            if not entries:
                continue
            first_metrics = next(iter(entries.values())) if isinstance(entries, dict) and entries else {}
            metric_defs = match_metric_configs(list(first_metrics.keys()) if isinstance(first_metrics, dict) else [])
            if not metric_defs:
                mk, dir_guess, label_guess = metric_from_keys(list(first_metrics.keys()) if isinstance(first_metrics, dict) else [])
                if mk:
                    metric_defs = [custom_metric_def(mk, dir_guess, label_guess)]
            for metric_def in metric_defs:
                source_key = metric_def.get("source")
                if not source_key:
                    continue
                ensure_metric_style(metric_def)
                recs = []
                for user_key, metrics in entries.items():
                    if not isinstance(metrics, dict):
                        continue
                    raw_val = metrics.get(source_key)
                    try:
                        v = float(raw_val)
                    except Exception:
                        continue
                    name = str(user_key)
                    base, submission_id = split_user_and_submission(name)
                    recs.append({
                        "user": base,
                        "raw_user": name,
                        "submission_id": submission_id,
                        "score": v,
                    })
                if not recs:
                    continue
                tmp = pd.DataFrame(recs)
                tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce")
                tmp = tmp.dropna(subset=["score"]).reset_index(drop=True)
                if tmp.empty:
                    continue
                direction = metric_def.get("direction") or "lower"
                tmp["when"] = resolve_submission_times(tmp, leaderboard_meta, when)
                tmp = tmp.drop(columns=["raw_user"], errors="ignore")
                tmp["label"] = metric_def.get("label", metric_def.get("title", "Score"))
                tmp["metric_key"] = metric_def.get("key")
                tmp["metric_label"] = metric_def.get("label", metric_def.get("title", "Score"))
                tmp["direction"] = direction
                tmp["phase"] = phase_name
                idx = tmp["score"].idxmax() if direction == "higher" else tmp["score"].idxmin()
                tmp["is_best"] = False
                if idx is not None and idx in tmp.index:
                    tmp.loc[idx, "is_best"] = True
                rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=[
            "when",
            "user",
            "score",
            "label",
            "direction",
            "is_best",
            "metric_key",
            "metric_label",
        ])
    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df.sort_values("when")
    return all_df


def compute_best_line(sub: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Return per-day best scores along with the running best for plotting."""
    if sub.empty:
        return pd.DataFrame(columns=[
            "day", "day_best_time", "day_best_score", "day_best_user",
            "best_score", "best_user", "best_time", "improved", "summary",
        ])

    work = sub.sort_values("when").copy()
    work["day"] = work["when"].dt.floor("D")

    try:
        group = work.groupby("day")["score"]
        idxs = group.idxmax() if direction == "higher" else group.idxmin()
    except Exception:
        return pd.DataFrame(columns=[
            "day", "day_best_time", "day_best_score", "day_best_user",
            "best_score", "best_user", "best_time", "improved", "summary",
        ])

    daily = work.loc[idxs].copy()
    if daily.empty:
        return pd.DataFrame(columns=[
            "day", "day_best_time", "day_best_score", "day_best_user",
            "best_score", "best_user", "best_time", "improved", "summary",
        ])

    daily = daily.sort_values("day").reset_index(drop=True)
    daily = daily.rename(columns={
        "when": "day_best_time",
        "score": "day_best_score",
        "user": "day_best_user",
    })

    best_scores: list[float] = []
    best_users: list[str] = []
    best_times: list[datetime] = []
    improved_flags: list[bool] = []
    summaries: list[str] = []

    best_score: float | None = None
    best_user: str | None = None
    best_time: datetime | None = None

    for row in daily.itertuples(index=False):
        score = row.day_best_score
        user = row.day_best_user
        when = row.day_best_time
        day = row.day
        improved = False
        if best_score is None:
            best_score = score
            best_user = user
            best_time = when
            improved = True
        elif direction == "higher":
            if score > best_score:
                best_score = score
                best_user = user
                best_time = when
                improved = True
        else:
            if score < best_score:
                best_score = score
                best_user = user
                best_time = when
                improved = True

        best_scores.append(best_score)
        best_users.append(best_user or "")
        best_times.append(best_time)
        improved_flags.append(improved)

        # Human-readable summary for hover text
        if when is None:
            when = day
        try:
            when_str = when.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            when_str = str(when)
        try:
            daily_score_str = f"{score:.5f}"
        except Exception:
            daily_score_str = "—"
        daily_user_str = str(user) if user else "—"
        if improved:
            prefix = "New best! "
        else:
            prefix = "Best held. "
        summaries.append(
            prefix + f"Daily best: {daily_user_str} at {when_str} ({daily_score_str})"
        )

    daily["best_score"] = best_scores
    daily["best_user"] = best_users
    daily["best_time"] = best_times
    daily["improved"] = improved_flags
    daily["summary"] = summaries

    return daily


def build_html_from_df(df: pd.DataFrame) -> str:
    if df.empty:
        return (
            '<div><p style="color:#666">No local Codabench results found in '
            'codabench_results/. Run get_leaderboard.py first.</p></div>'
        )
    last = df["when"].max()
    pieces: list[str] = [
        f"<div>\n  <p style=\"color:#666; margin-bottom:8px;\">Last updated: {last.strftime('%Y-%m-%dT%H:%M:%SZ')} (UTC)</p>"
    ]

    # Group by phase and render a figure for each
    phases_from_df = df["phase"].dropna().unique().tolist() if "phase" in df.columns else []
    # Merge with names seen in summaries (preserve insertion order)
    def uniq_preserve(seq):
        seen = set(); out = []
        for x in seq:
            if x is None: 
                continue
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    phases = uniq_preserve(phases_from_df)
    for name in PHASE_NAMES_SEEN:
        if name not in phases:
            phases.append(name)
    if not phases:
        phases = [None]
    # Order with Warmup first, Final last
    def phase_rank(n):
        if n is None:
            return 1
        s = n.lower()
        if ("warm" in s) or ("phase 1" in s) or ("development" in s) or ("dev" in s):
            return 0
        if ("final" in s) or ("phase 2" in s):
            return 2
        return 1
    phases = sorted(phases, key=lambda n: (phase_rank(n), str(n).lower()))
    for idx, ph in enumerate(phases, start=1):
        sub = df[df["phase"].eq(ph)] if ph is not None else df
        title_text = f"{TITLE_PREFIX} - {ph}" if ph else TITLE_PREFIX
        axis_label = "Challenge Scores"

        traces: list[go.Scatter] = []
        axis_scores: dict[str, list[float]] = {"y": [], "y2": []}
        metric_payloads: list[dict[str, Any]] = []
        if not sub.empty:
            metric_keys = sub["metric_key"].dropna().unique().tolist() if "metric_key" in sub.columns else []
            if not metric_keys:
                metric_keys = [None]

            def metric_sort_key(k: str | None) -> tuple[int, str]:
                if k is None:
                    return (99, "")
                style = METRIC_STYLE_LOOKUP.get(k)
                return (style.get("order", 99) if style else 99, str(k))

            metric_keys = sorted(metric_keys, key=metric_sort_key)

            for metric_key in metric_keys:
                metric_df = sub[sub["metric_key"].eq(metric_key)] if metric_key is not None else sub
                if metric_df.empty:
                    continue
                metric_label = metric_df["metric_label"].iloc[0] if "metric_label" in metric_df.columns else (OVR_LABEL or "Score")
                direction = metric_df["direction"].iloc[0] if "direction" in metric_df.columns else "lower"
                style = style_for_metric(metric_key or "overall", metric_label, direction)

                axis_id = "y2" if (metric_key and "overall" in str(metric_key).lower()) or ("combined" in metric_label.lower()) else "y"
                axis_scores.setdefault(axis_id, []).extend(metric_df["score"].dropna().tolist())
                metric_payloads.append({
                    "metric_key": metric_key,
                    "metric_label": metric_label,
                    "direction": direction,
                    "style": style,
                    "axis_id": axis_id,
                    "df": metric_df.copy(),
                })
        y_info = compute_axis_range(pd.Series(axis_scores.get("y")))
        y2_info = compute_axis_range(pd.Series(axis_scores.get("y2")))
        if y_info:
            y_range, y_cap = y_info
        else:
            y_range, y_cap = (None, None)
        if y2_info:
            y2_range, y2_cap = y2_info
        else:
            y2_range, y2_cap = (None, None)
        if not axis_scores.get("y"):
            axis_label = "Score"

        for payload in metric_payloads:
            metric_df = payload["df"].copy()
            metric_label = payload["metric_label"]
            direction = payload["direction"]
            style = payload["style"]
            axis_id = payload["axis_id"]
            legend_group = payload["metric_key"] or metric_label

            cap = y_cap if axis_id == "y" else y2_cap
            metric_df["plot_score"] = metric_df["score"]
            if cap is not None:
                metric_df["plot_score"] = metric_df["plot_score"].clip(upper=cap)

            scatter = go.Scatter(
                x=metric_df["when"],
                y=metric_df["plot_score"],
                text=metric_df["user"],
                mode="markers",
                name=f"{metric_label} models",
                legendgroup=legend_group,
                yaxis=axis_id,
                marker=dict(color=style.get("scatter_color", "rgba(140,140,140,0.35)"), size=11),
                hovertemplate=(
                    "%{text}<br>%{x|%Y-%m-%d %H:%M UTC}<br>"
                    + metric_label
                    + ": %{customdata[0]:.5f}<extra></extra>"
                ),
                customdata=metric_df[["score"]].to_numpy(),
                cliponaxis=False,
            )
            traces.append(scatter)

            best_df = compute_best_line(metric_df.assign(score=metric_df["score"]), direction)
            if best_df.empty:
                continue
            best_df = best_df[best_df["improved"]].copy()
            if best_df.empty:
                continue
            best_df["plot_score"] = best_df["best_score"]
            if cap is not None:
                best_df["plot_score"] = best_df["plot_score"].clip(upper=cap)
            best_df["best_time_str"] = best_df["best_time"].dt.strftime("%Y-%m-%d %H:%M UTC")
            best_df["day_time_str"] = best_df["day_best_time"].dt.strftime("%Y-%m-%d %H:%M UTC")
            marker_text = best_df["best_user"].tolist()
            customdata = [
                [
                    best_user,
                    best_time_str,
                    day_user,
                    day_time_str,
                    (f"{day_score:.5f}" if pd.notna(day_score) else "—"),
                    summary,
                    original,
                ]
                for best_user, best_time_str, day_user, day_time_str, day_score, summary, original in zip(
                    best_df["best_user"],
                    best_df["best_time_str"],
                    best_df["day_best_user"],
                    best_df["day_time_str"],
                    best_df["day_best_score"],
                    best_df["summary"],
                    best_df["best_score"],
                )
            ]
            best_trace = go.Scatter(
                x=best_df["day_best_time"],
                y=best_df["plot_score"],
                text=marker_text,
                mode="lines+markers+text",
                name=f"Best {metric_label}",
                legendgroup=legend_group,
                yaxis=axis_id,
                marker=dict(
                    color=style.get("line_color", ACCENT),
                    size=16,
                    line=dict(width=2, color=style.get("line_dark", ACCENT_DARK)),
                ),
                line=dict(color=style.get("line_color", ACCENT), width=3),
                textposition=style.get("best_text_position", "top center"),
                textfont=dict(size=13, color="#333"),
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[5]}<br>"
                    "Best so far: %{customdata[0]} (since %{customdata[1]})<br>"
                    + "Best " + metric_label + ": %{customdata[6]:.5f}<extra></extra>"
                ),
                line_shape="hv",
                cliponaxis=False,
            )
            traces.append(best_trace)

        x_series = sub["when"].dropna()
        x_range = None
        if not x_series.empty:
            date_min = x_series.min()
            date_max = x_series.max()
            if date_min == date_max:
                x_range = [date_min - pd.Timedelta(days=1), date_max + pd.Timedelta(days=1)]
            else:
                span = date_max - date_min
                pad = span * 0.05
                x_range = [date_min - pad, date_max + pad]

        layout = go.Layout(
            title=dict(text=title_text, x=0.5, font=dict(size=24, color="#2f2f2f")),
            margin=dict(l=70, r=20, t=70, b=80),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(
                title=dict(text="Snapshot Time (UTC)", font=dict(size=16, color="#334155")),
                type="date", showgrid=True,
                gridcolor="#e9e9e9", gridwidth=1, zeroline=False,
                tickfont=dict(size=14, color="#3a3a3a"),
                range=x_range,
            ),
            yaxis=dict(
                title=dict(text=axis_label, font=dict(size=16, color="#334155")),
                type="linear", autorange=True, showgrid=True,
                gridcolor="#e9e9e9", gridwidth=1, zeroline=False,
                tickfont=dict(size=14, color="#3a3a3a"),
                range=y_range,
            ),
            yaxis2=dict(
                title=dict(text="Combined Score", font=dict(size=16, color="#334155")),
                type="linear", autorange=True, showgrid=False,
                overlaying="y", side="right",
                tickfont=dict(size=13, color="#3a3a3a"),
                range=y2_range,
            ),
            legend=dict(orientation="h", x=0.15, y=-0.18, font=dict(size=13, color="#3a3a3a")),
            hovermode="closest",
            height=560,
        )
        if not axis_scores.get("y2"):
            layout.update(yaxis2=dict(overlaying="y", side="right", showgrid=False, visible=False, showticklabels=False))
        # Apply per-phase x-axis range, hard-coded for Warmup/Final
        xr = phase_xrange(ph)
        if xr:
            layout.update(xaxis=dict(title=dict(text="Snapshot Time (UTC)", font=dict(size=16, color="#334155")), type="date", showgrid=True, range=list(xr), gridcolor="#e9e9e9", gridwidth=1, zeroline=False, tickfont=dict(size=14, color="#3a3a3a")))
        elif X_START and X_END:
            layout.update(xaxis=dict(title=dict(text="Snapshot Time (UTC)", font=dict(size=16, color="#334155")), type="date", showgrid=True, range=[X_START, X_END], gridcolor="#e9e9e9", gridwidth=1, zeroline=False, tickfont=dict(size=14, color="#3a3a3a")))

        fig = go.Figure(data=traces, layout=layout)
        if not traces:
            # Add a subtle placeholder annotation when no data yet
            fig.add_annotation(text="No data yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="#888", size=14))
        div_id = f"leaderboard-plot-{idx}"
        plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id=div_id, config={"displaylogo": False})
        pieces.append("  " + plot_html)
        pieces.append("  <div style=\"height:12px\"></div>")

    pieces.append("</div>")
    return "\n".join(pieces)


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    df = load_dataframe()
    html = build_html_from_df(df)
    DEST.write_text(html)
    n_snapshots = df["when"].nunique() if not df.empty else 0
    print(f"Wrote {DEST} with {n_snapshots} snapshot(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
