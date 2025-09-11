import os
import re
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent.resolve()
CB_DIR = Path(os.environ.get("CB_OUT", ROOT / "codabench_results"))
DEST = ROOT / "_includes" / "leaderboard_generated.html"

# Optional overrides via environment
OVR_REGEX = os.environ.get("LB_METRIC_REGEX", r"(?i)(normalized|normalised)\s*mse|\bnmse\b|\bmse\b|overall score$")
OVR_LABEL = os.environ.get("LB_LABEL", "Normalize MSE")
OVR_DIRECTION = os.environ.get("LB_DIRECTION")  # 'lower' or 'higher'
PHASE_REGEX = os.environ.get("LB_PHASE_REGEX", r"(?i)(warm\s*up|warmup|development|dev|phase\s*1)")
X_START = os.environ.get("LB_X_START")  # e.g., '2025-09-01' (UTC)
X_END = os.environ.get("LB_X_END")      # e.g., '2025-10-03' (UTC)
ACCENT = os.environ.get("LB_ACCENT", "#F2994A")
ACCENT_DARK = os.environ.get("LB_ACCENT_DARK", "#D97E2E")
TITLE_PREFIX = os.environ.get("LB_TITLE_PREFIX", "Leaderboard")

# Track phase names seen so we can render a figure even if a phase has no data
PHASE_NAMES_SEEN: set[str] = set()


def parse_time(s: str | None, fallback: float) -> datetime:
    if s:
        try:
            if s.endswith("Z"):
                dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
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
        out.append({"name": name, "csv": csv_path, "json": json_path})
    # If none, try merged file fallback (rare)
    if not out:
        merged = summary.get("merged_results_json")
        if merged:
            mp = Path(merged)
            if not mp.is_absolute():
                mp = (ROOT / mp).resolve()
            if mp.exists():
                out.append({"name": "", "csv": None, "json": mp})
    return out


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
                    mcol, direction, label = select_metric_column(df)
                    if mcol:
                        tmp = df[[user_col, mcol]].copy()
                        tmp.columns = ["user", "score"]
                        tmp["user"] = tmp["user"].astype(str).str.replace(r"-\d+$", "", regex=True)
                        tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce")
                        tmp = tmp.dropna(subset=["score"]).reset_index(drop=True)
                        if not tmp.empty:
                            tmp["when"] = when
                            tmp["label"] = label
                            tmp["direction"] = direction
                            tmp["phase"] = phase_name
                            # best per snapshot within this phase
                            idx = tmp["score"].idxmax() if direction == "higher" else tmp["score"].idxmin()
                            tmp["is_best"] = False
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
            mkey, direction, label = metric_from_keys(list(first_metrics.keys()))
            if not mkey:
                continue
            recs = []
            for user_key, metrics in entries.items():
                try:
                    v = float(metrics.get(mkey))
                except Exception:
                    continue
                name = str(user_key)
                if "-" in name:
                    parts = name.split("-")
                    if parts[-1].isdigit():
                        name = "-".join(parts[:-1])
                recs.append({"user": name, "score": v})
            if not recs:
                continue
            tmp = pd.DataFrame(recs)
            tmp["when"] = when
            tmp["label"] = label
            tmp["direction"] = direction
            tmp["phase"] = phase_name
            idx = tmp["score"].idxmax() if direction == "higher" else tmp["score"].idxmin()
            tmp["is_best"] = False
            tmp.loc[idx, "is_best"] = True
            rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["when", "user", "score", "label", "direction", "is_best"])
    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df.sort_values("when")
    return all_df


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
    # Union with names seen in summaries to ensure empty phases also render
    phases = list({*(phases_from_df or []), *PHASE_NAMES_SEEN}) or [None]
    for idx, ph in enumerate(phases, start=1):
        sub = df[df["phase"].eq(ph)] if ph is not None else df
        label = (sub["label"].iloc[0] if (not sub.empty and "label" in sub.columns) else (OVR_LABEL or "Score"))
        title_text = f"{TITLE_PREFIX} - {ph}" if ph else TITLE_PREFIX

        accent = ACCENT
        accent_dark = ACCENT_DARK
        grey_pts = "rgba(140,140,140,0.38)"

        traces = []
        if not sub.empty:
            all_trace = go.Scatter(
                x=sub["when"], y=sub["score"], text=sub["user"], mode="markers",
                name="All models", marker=dict(color=grey_pts, size=12),
                hovertemplate="%{text}<br>%{x|%Y-%m-%d %H:%M UTC}<br>" + label + ": %{y:.5f}<extra></extra>"
            )
            traces.append(all_trace)
            # Best-of-day line: pick best score for each date
            direction = sub["direction"].iloc[0] if "direction" in sub.columns else "lower"
            day = sub["when"].dt.normalize()
            sub2 = sub.copy()
            sub2["day"] = day
            if direction == "higher":
                idxs = sub2.groupby("day")["score"].idxmax()
            else:
                idxs = sub2.groupby("day")["score"].idxmin()
            best_df = sub2.loc[idxs].sort_values("day")
            best_trace = go.Scatter(
                x=best_df["day"], y=best_df["score"], text=best_df["user"],
                mode="lines+markers+text", name="Best Model",
                marker=dict(color=accent, size=14, line=dict(width=2, color=accent_dark)),
                line=dict(color=accent, width=3), textposition="top center",
                textfont=dict(size=13, color="#333"),
                hovertemplate="%{text}<br>%{x|%Y-%m-%d %H:%M UTC}<br>Best " + label + ": %{y:.5f}<extra></extra>"
            )
            traces.append(best_trace)
        layout = go.Layout(
            title=dict(text=title_text, x=0.5, font=dict(size=24, color="#2f2f2f")),
            margin=dict(l=70, r=20, t=70, b=80),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(
                title=dict(text="Snapshot Time (UTC)", font=dict(size=16, color="#334155")),
                type="date", showgrid=True,
                gridcolor="#e9e9e9", gridwidth=1, zeroline=False,
                tickfont=dict(size=14, color="#3a3a3a")
            ),
            yaxis=dict(
                title=dict(text=label, font=dict(size=16, color="#334155")),
                autorange=True, showgrid=True,
                gridcolor="#e9e9e9", gridwidth=1, zeroline=False,
                tickfont=dict(size=14, color="#3a3a3a")
            ),
            legend=dict(orientation="h", x=0.15, y=-0.18, font=dict(size=13, color="#3a3a3a")),
            hovermode="closest",
            height=560,
        )
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
