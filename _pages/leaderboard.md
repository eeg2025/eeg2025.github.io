---
layout: page
permalink: /leaderboard/
title: Leaderboard
description: Interactive leaderboard summary from Codabench results.
nav: true
nav_order: 7
---

<div>
  <p id="lb-updated" style="color:#666; margin-bottom:8px;">Loading leaderboard…</p>
  <div id="leaderboard-plot" style="width:100%;max-width:1100px;height:560px;"></div>
  <p style="color:#666; font-size:0.95em;">
    Source: local snapshot(s) in <code>codabench_results/</code>.
    Latest data is pulled from <a href="https://www.codabench.org/competitions/9975/#/results-tab" target="_blank" rel="noopener">Codabench</a> by the repository script.
  </p>
</div>

<script src="https://cdn.plot.ly/plotly-2.34.0.min.js"></script>
<script>
// Helper: site base path for correct links on GitHub Pages
const BASEURL = "{{ site.baseurl | default: '' }}";
function toUrl(p) {
  if (!p) return null;
  if (p.startsWith('http://') || p.startsWith('https://')) return p;
  if (p.startsWith('/')) return BASEURL + p;
  return BASEURL + '/' + p;
}

// Collect available summary.json files under codabench_results.
// If there are multiple dated folders with a summary.json each, we will combine them as a time series.
const SUMMARY_PATHS = [
  // Auto-discover via Jekyll: includes root file and any nested dated folders
  {% assign summaries = site.static_files | where_exp: 'f', 'f.path contains "/codabench_results/"' %}
  {% for f in summaries %}{% if f.path contains 'summary.json' %}"{{ f.path }}",{% endif %}{% endfor %}
].filter((v, i, a) => v && a.indexOf(v) === i);

if (SUMMARY_PATHS.length === 0) {
  // Fallback to the conventional location
  SUMMARY_PATHS.push('codabench_results/summary.json');
}

async function fetchJSON(url) {
  const r = await fetch(url, {cache: 'no-cache'});
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
  return await r.json();
}

function parseUsername(keyWithId) {
  if (!keyWithId) return keyWithId;
  const parts = String(keyWithId).split('-');
  if (parts.length <= 1) return keyWithId;
  const id = parts.pop();
  return parts.join('-');
}

function pickMetricKeyAndDirection(obj) {
  const keys = Object.keys(obj || {});
  if (!keys.length) return { key: null, direction: 'lower', label: 'Normalized MSE' };
  // Priority: Normalized MSE (or NMSE) → MSE → overall score → first key
  const nmse = keys.find(k => /(normalized|normalised)?\s*mse\b/i.test(k))
            || keys.find(k => /\bnmse\b/i.test(k));
  if (nmse) return { key: nmse, direction: 'lower', label: 'Normalized MSE' };
  const mse = keys.find(k => /\bmse\b/i.test(k));
  if (mse) return { key: mse, direction: 'lower', label: 'MSE' };
  const overall = keys.find(k => /overall score$/i.test(k));
  if (overall) return { key: overall, direction: 'higher', label: 'Score' };
  return { key: keys[0], direction: 'higher', label: 'Score' };
}

function extractPhaseBlock(resultsObj) {
  // resultsObj is expected to have top-level keys like "Results - Warmup Phase(16261)"
  const keys = Object.keys(resultsObj || {});
  if (keys.length === 0) return { name: 'N/A', entries: {} };
  // Prefer Final Phase if available and non-empty
  const finalKey = keys.find(k => /final/i.test(k));
  if (finalKey && resultsObj[finalKey] && Object.keys(resultsObj[finalKey]).length) {
    return { name: finalKey, entries: resultsObj[finalKey] };
  }
  // Otherwise use the first non-empty block
  const firstKey = keys.find(k => resultsObj[k] && Object.keys(resultsObj[k]).length);
  return { name: firstKey || keys[0], entries: resultsObj[firstKey || keys[0]] || {} };
}

function detectDirection(values) {
  // Heuristic: if values are all within [0,1.00001], treat higher-is-better; otherwise lower-is-better
  if (!values.length) return 'lower';
  const max = Math.max(...values);
  return (max <= 1.00001) ? 'higher' : 'lower';
}

async function loadSnapshots(summaryPaths) {
  const snapshots = [];
  for (const sp of summaryPaths) {
    const url = toUrl(sp);
    try {
      const summary = await fetchJSON(url);
      const when = new Date(summary.generated_at_utc || summary.generated_at || Date.now());
      let resPath = summary.merged_results_json || (summary.phases && summary.phases[0] && summary.phases[0].json);
      if (!resPath) continue;
      resPath = toUrl(resPath);
      const results = await fetchJSON(resPath);
      const { name: phaseName, entries } = extractPhaseBlock(results);
      const scores = [];
      let label = 'Score';
      let directionHint = 'higher';
      const entriesArr = Object.entries(entries || {});
      let chosenKey = null;
      if (entriesArr.length) {
        const firstSel = pickMetricKeyAndDirection(entriesArr[0][1] || {});
        chosenKey = firstSel.key;
        label = firstSel.label || label;
        directionHint = firstSel.direction || directionHint;
      }
      for (const [userKey, metrics] of entriesArr) {
        const key = chosenKey || pickMetricKeyAndDirection(metrics).key;
        if (!key) continue;
        const value = parseFloat(metrics[key]);
        if (Number.isFinite(value)) {
          scores.push({ user: parseUsername(userKey), raw: userKey, score: value });
        }
      }
      if (!scores.length) continue;
      const direction = directionHint || detectDirection(scores.map(s => s.score));
      const best = scores.reduce((acc, s) => {
        if (!acc) return s;
        if (direction === 'higher') return (s.score > acc.score) ? s : acc;
        return (s.score < acc.score) ? s : acc;
      }, null);
      snapshots.push({ when, phaseName, scores, best, direction, label });
    } catch (e) {
      console.warn('Skipping snapshot due to error:', e);
    }
  }
  // Sort by time ascending
  snapshots.sort((a, b) => a.when - b.when);
  return snapshots;
}

function renderPlot(snapshots) {
  const updatedEl = document.getElementById('lb-updated');
  if (!snapshots.length) {
    updatedEl.textContent = 'No local Codabench results found in codabench_results/.';
    return;
  }
  const lastDate = snapshots[snapshots.length - 1].when;
  updatedEl.textContent = `Last updated: ${lastDate.toLocaleString()} (${snapshots.length} snapshot${snapshots.length>1?'s':''})`;

  // Build scatter of all models across snapshots
  const allX = [], allY = [], allText = [], allMeta = [];
  let direction = 'lower';
  let yLabel = snapshots[0]?.label || 'Normalized MSE';
  for (const snap of snapshots) {
    direction = snap.direction || direction;
    if (snap.label) yLabel = snap.label;
    for (const s of snap.scores) {
      allX.push(snap.when);
      allY.push(s.score);
      allText.push(`${s.user}`);
      allMeta.push({ user: s.user, when: snap.when, score: s.score });
    }
  }

  const allTrace = {
    x: allX,
    y: allY,
    text: allText,
    mode: 'markers',
    name: 'All models',
    marker: { color: 'rgba(120,120,120,0.35)', size: 9 },
    hovertemplate: `%{text}<br>%{x|%Y-%m-%d %H:%M UTC}<br>${yLabel}: %{y:.5f}<extra></extra>`
  };

  // Best per snapshot (line)
  const bestX = [], bestY = [], bestText = [];
  for (const snap of snapshots) {
    bestX.push(snap.when);
    bestY.push(snap.best.score);
    bestText.push(snap.best.user);
  }
  const bestTrace = {
    x: bestX,
    y: bestY,
    text: bestText,
    mode: 'lines+markers+text',
    name: 'Best per snapshot',
    marker: { color: '#6b5ca5', size: 10, line: { width: 1, color: '#51457f' } },
    line: { color: '#6b5ca5', width: 2 },
    textposition: 'top center',
    textfont: { size: 12, color: '#333' },
    hovertemplate: `%{text}<br>%{x|%Y-%m-%d %H:%M UTC}<br>Best ${yLabel}: %{y:.5f}<extra></extra>`
  };

  const yTitle = yLabel || 'Normalized MSE';
  const layout = {
    title: { text: 'Leaderboard', x: 0.5 },
    margin: { l: 60, r: 20, t: 60, b: 60 },
    xaxis: { title: 'Snapshot Time (UTC)', type: 'date', showgrid: true },
    yaxis: { title: yTitle, autorange: true, showgrid: true },
    legend: { orientation: 'h', x: 0.15, y: -0.2 },
    hovermode: 'closest'
  };

  Plotly.newPlot('leaderboard-plot', [allTrace, bestTrace], layout, {displaylogo: false});
}

(async function() {
  try {
    const snaps = await loadSnapshots(SUMMARY_PATHS);
    renderPlot(snaps);
  } catch (e) {
    document.getElementById('lb-updated').textContent = 'Failed to load leaderboard data.';
    console.error(e);
  }
})();
</script>
